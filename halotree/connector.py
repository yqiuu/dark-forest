import os
from abc import ABC, abstractmethod

from .logger import CreateLogger
from .halotree import *

import numpy as np
import pandas as pd
import h5py
from astropy.cosmology import FlatLambdaCDM
from hmf.hmf import MassFunction

__all__ = [
    'DataHandler',
    'VELOCIraptorHandler',
    'SaveHaloConnector',
    'ReadHaloConnector'
]


logger = CreateLogger(__name__)


class DataHandler(ABC):
    @abstractmethod
    def CacheSnapshot(self, snap):
        pass


    @abstractmethod
    def __getitem__(self, key):
        pass

    
    @abstractmethod
    def __enter__(self):
        pass


    @abstractmethod
    def __exit__(self, *args):
        pass


    @abstractmethod
    def __init__(self, fp):
        pass


class VELOCIraptorHandler(DataHandler):
    def HubbleConstant(self):
        return self.fp.get('Header')['Simulation'].attrs['h_val']


    def OmegaM(self):
        return self.fp.get('Header')['Simulation'].attrs['Omega_m']


    def HaloKit(self, **kwargs):
        params = {'Mmin':5, 'dlog10m':0.05}
        params.update(kwargs)
        params['cosmo_params'] = {'H0':100*self.HubbleConstant(), 'Om0':self.OmegaM()}
        # The following two parameters cannot be directly read from the output
        params['sigma_8'] = 0.815
        params['n'] = 0.9653
        hmf = MassFunction(**params)
        return HaloKit(hmf)


    def MassConversion(self):
        """
        Returns
        -------
        float:
            A mass conversion factor to 10^10 h^-1 M_solar.
        """
        return 1e10*self.HubbleConstant()


    def NSnap(self):
        n_snap = 0
        keys = list(self.fp.keys())
        for k in keys:
            if k.startswith('Snap'):
                n_snap += 1
        return n_snap


    def Redshifts(self, snap_max = -1):
        if snap_max == -1:
            n_snap = self.NSnap()
        else:
            n_snap = snap_max + 1
        redshifts = np.zeros(n_snap)
        for snap in range(n_snap):
            redshifts[snap] = 1./self.fp.get('Snap_%03d'%snap).attrs['scalefactor'] - 1.
        return redshifts


    def Ages(self, snap_max = -1):
        """
        Returns
        -------
        array [Myr]
            Age of the universe at each time step
        """
        cosmo = FlatLambdaCDM(H0 = 100*self.HubbleConstant(), Om0 = self.OmegaM())
        return cosmo.age(self.Redshifts(snap_max)).value*1e3


    def CacheSnapshot(self, snap):
        self.dataset = self.fp.get('Snap_%03d'%snap)


    def __getitem__(self, key):
        if key == 'DescendantSnap':
            return np.asarray(self.dataset['Descendant'])//10**12
        elif key == 'HostHaloID':
            return np.asarray(self.dataset['hostHaloID'])
        else:
            return np.asarray(self.dataset[key])


    def __enter__(self):
        self.fp = h5py.File(self.fname, "r")
        return self

    
    def __exit__(self, *args):
        self.fp.close()
        

    def __init__(self, fname):
        self.fname = fname


def SaveOutput(handler, data, mass_key, snap, path, prefix):
    n_prog = len(data)
    if n_prog == 0:
        logger.warning("no progenitor for halos at snapshot %d"%snap)
        return
    mass_unit = handler.MassConversion()
    columns = ['ID', 'Snap']
    columns.extend(mass_key)
    plain_data = []
    count = np.zeros(n_prog, dtype = int)
    for i_prog, progenitors in enumerate(data.values()):
        plain_data.extend(progenitors)
        count[i_prog] = len(progenitors)
    df_prog = pd.DataFrame(plain_data, columns = columns)
    df_prog[mass_key] *= mass_unit
    df_prog.set_index('ID', inplace = True)
    df_prog.to_hdf(os.path.join(path, '%s_prog_%d.hdf5'%(prefix, snap)), "w")
    #
    count_cum = np.cumsum(count)
    df_host = pd.DataFrame()
    df_host['IndexStart'] = np.append(0, count_cum[:-1])
    df_host['IndexEnd'] = count_cum
    df_host['Number'] = count
    df_host['ID'] = list(data.keys())
    df_host.set_index('ID', inplace = True)
    halo_id = handler['ID']
    inds = np.asarray(pd.Series(np.arange(len(halo_id)), index = halo_id)[df_host.index])
    for k in mass_key:
        df_host[k] = handler[k][inds]
    df_host[mass_key] *= mass_unit
    df_host.to_hdf(os.path.join(path, '%s_host_%d.hdf5'%(prefix, snap)), "w")


def SaveHaloConnector(handler, snap_min, snap_max, mass_key, path = './', prefix = 'hc'):
    mass_key = np.ravel(mass_key)
    n_snap = snap_max + 1
    desc_pool_arr = np.asarray([{} for snap in range(n_snap)])
    for snap in range(snap_min, n_snap):
        logger.info("Processing snaphost %d"%snap)
        handler.CacheSnapshot(snap)
        # Merge subhalos to their host halo
        desc_pool = desc_pool_arr[snap]
        halo_id_arr = handler['ID']
        host_id_arr = handler['HostHaloID']
        host_table = dict(zip(halo_id_arr, host_id_arr))
        for halo_id in list(desc_pool.keys()):
            host_id = host_table[halo_id]
            progenitors = desc_pool[halo_id]
            if host_id != -1:
                try:
                    desc_pool[host_id].extend(progenitors)
                except KeyError:
                    desc_pool[host_id] = progenitors
                del desc_pool[halo_id]
        # Save existing descendants
        SaveOutput(handler, desc_pool, mass_key, snap, path, prefix)
        # Walk forward
        desc_id_arr = handler['Descendant']
        desc_snap_arr = handler['DescendantSnap']
        props_arr = np.asarray([handler[k] for k in mass_key])
        for desc_id, desc_snap, host_id, halo_id, *props in zip(
            desc_id_arr, desc_snap_arr, host_id_arr, halo_id_arr, *props_arr
        ):
            if host_id == -1 and desc_snap <= snap_max:
                target_pool = desc_pool_arr[desc_snap]
                try:
                    target_pool[desc_id].append((halo_id, snap, *props))
                except KeyError:
                    target_pool[desc_id] = [(halo_id, snap, *props)]
        desc_pool_arr[snap] = None


def ReadHaloConnector(handler, indices, snap_max, snap_min, mass_key, path, prefix):
    indices = np.ravel(indices)
    # Reselect host halos
    n_in = len(indices)
    handler.CacheSnapshot(snap_max)
    host_id = handler['HostHaloID'][indices]
    indices = indices[host_id == -1]
    n_gal = len(indices)
    logger.info("Fraction of host halos %d/%d"%(n_gal, n_in))
    #
    redshifts = handler.Redshifts(snap_max)
    # Initialise root halos and the pool to read progenitors
    z_root = redshifts[snap_max]
    m_root_arr = handler[mass_key][indices]
    halo_id_arr = handler['ID'][indices]
    tree_arr = np.empty(n_gal, dtype = object)
    pool = [[]]*(snap_max + 1)
    for i_tree, (halo_id, m_root) in enumerate(zip(halo_id_arr, m_root_arr)):
        tree = HaloMergeTree(method = 'empty', n_base = 10**4)
        tree.SetRedshifts(redshifts)
        tree.Append(z_root, m_root)
        tree_arr[i_tree] = tree
        pool[snap_max].append((i_tree, 0, halo_id))
    # Read Progenitors
    for snap in range(snap_max, snap_min - 1, -1):
        logger.info("Read snapshot %d"%snap)
        z = redshifts[snap - 1]
        prog_data = pd.read_hdf(os.path.join(path, '%s_prog_%d.hdf5'%(prefix, snap))).to_records()
        host_info = pd.read_hdf(os.path.join(path, '%s_host_%d.hdf5'%(prefix, snap)))
        prog_start = host_info['IndexStart']
        prog_end = host_info['IndexEnd']
        prog_dict = dict(zip(host_info.index, zip(prog_start, prog_end)))
        for i_tree, idx_node_host, host_id in pool[snap]:
            if host_id in prog_dict:
                tree = tree_arr[i_tree]
                start, end = prog_dict[host_id]
                progenitors = prog_data[start:end]
                for i_prog, prog in enumerate(progenitors):
                    halo_id = prog['ID']
                    halo_snap = prog['Snap']
                    mass = prog[mass_key]
                    idx_node_new = tree.Append(z, mass)
                    if i_prog == 0:
                        # Append first progenitor
                        tree.LinkTailIndex(idx_node_host, idx_node_new)
                    else:
                        # Append next progenitor
                        tree.LinkNextIndex(idx_node_prev, idx_node_new)
                    pool[halo_snap].append((i_tree, idx_node_new, halo_id))
                    idx_node_prev = idx_node_new
        pool[snap] = None
    return tree_arr
