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


    def Volume(self):
        boxsize = self.fp.get('Header')['Simulation'].attrs['Period']
        return boxsize*boxsize*boxsize


    def HaloKit(self, **kwargs):
        params = {'Mmin':5, 'dlog10m':0.05}
        params.update(kwargs)
        params['cosmo_params'] = {'H0':100*self.HubbleConstant(), 'Om0':self.OmegaM()}
        # The following two parameters cannot be directly read from the output
        params['sigma_8'] = 0.815
        params['n'] = 0.9653
        hmf = MassFunction(**params)
        return HaloKit(hmf), hmf


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


def SaveOutput(handler, prog_data, mass_key, snap, path, prefix):
    mass_unit = handler.MassConversion()
    # Select all host halos at the current snapshot
    cond = handler['HostHaloID'] == -1
    halo_id = handler['ID'][cond]
    df_host = pd.DataFrame(halo_id, columns = ['ID'])
    for key in np.append(['Descendant', 'DescendantSnap'], mass_key):
        df_host[key] = handler[key][cond]
    df_host[mass_key] *= mass_unit
    df_host.set_index('ID', inplace = True)
    # Prepare progenitor data
    n_prog = len(prog_data)
    plain_data = []
    number = np.zeros(n_prog, dtype = int)
    for i_prog, prog in enumerate(prog_data.values()):
        number[i_prog] = len(prog)
        plain_data.extend(prog)
    df_prog = pd.DataFrame(plain_data, columns = ['ID', 'Snap'] + [k for k in mass_key])
    df_prog[mass_key] *= mass_unit
    df_prog.set_index('ID', inplace = True)
    # Set progenitor info
    df_host.insert(0, 'ProgIndex', -1)
    df_host.insert(1, 'ProgNumber', 0)
    df_host.loc[prog_data.keys(), 'ProgIndex'] = np.append(0, np.cumsum(number)[:-1])
    df_host.loc[prog_data.keys(), 'ProgNumber'] = number
    # Save data
    with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap)), "w") as store:
        store.append('Prog', df_prog)
        store.append('Host', df_host)


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


def ReadHaloConnector(path, prefix, indices, snap_min, snap_max, mass_key, redshifts):
    # Read Progenitors
    for snap in range(snap_max, snap_min - 1, -1):
        logger.info("Read snapshot %d"%snap)
        z = redshifts[snap - 1]
        # Read data
        with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap))) as store:
            prog_data = pd.read_hdf(store, 'Prog', columns = ['Snap', mass_key]).to_records()
            host_data = pd.read_hdf(store, 'Host', columns = ['ProgIndex', 'ProgNumber', mass_key])
        if snap == snap_max:
            # Initialise root halos and the pool to read progenitors
            z_root = redshifts[snap_max]
            indices = np.ravel(indices)
            halo_id_arr = host_data.index[indices]
            m_root_arr = host_data.loc[halo_id_arr, mass_key]
            tree_arr = np.empty(len(indices), dtype = object)
            pool = [[] for i_snap in range(snap_max + 1)]
            for i_tree, (halo_id, m_root) in enumerate(zip(halo_id_arr, m_root_arr)):
                tree = HaloMergeTree(method = 'empty', n_base = 10**4)
                tree.SetRedshifts(redshifts)
                tree.Append(z_root, m_root)
                tree_arr[i_tree] = tree
                pool[snap_max].append((i_tree, 0, halo_id))
        # Trace progenitors
        current_pool = np.asarray(pool[snap])
        if len(current_pool) == 0:
            continue
        prog_pointer = host_data.loc[current_pool[:, -1], ['ProgIndex', 'ProgNumber']].values
        cond = prog_pointer[:, -1] > 0
        current_pool = current_pool[cond]
        prog_pointer = prog_pointer[cond]
        for (i_tree, idx_node_host, host_id), (idx, num) in zip(current_pool, prog_pointer):
            tree = tree_arr[i_tree]
            for i_prog, prog in enumerate(prog_data[idx:idx + num]):
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
