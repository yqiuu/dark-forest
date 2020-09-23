import os

from .._dtree import MergerTree
from .connector import Connector

import tables
import numpy as np
import pandas as pd
from tables.exceptions import NoSuchNodeError
from tqdm import trange
from astropy import units as u


__all__ = [
    'DarkForestConnector',
    'DarkForestSaver'
]


class DarkForestConnector(Connector):
    def __init__(self, fname, prefix=None, write=False, filters=None):
        self.prefix = prefix
        if prefix is None:
            self.file_names = [fname]
            self.targets = [fname]
            self.i_part = 0
            self.write = write
            self.filters = filters
        else:
            file_names = []
            for f in os.listdir(fname):
                if f.startswith(prefix):
                    file_names.append(os.path.join(fname, f))
            file_names.sort()
            self.file_names = file_names
            if write is False:
                self.targets = list(file_names)
                self.i_part = 0
                self.write = False
                self.filters = None
            else:
                self.targets = [file_names[write]]
                self.i_part = write
                self.write = True
                self.filters = filters


    def __enter__(self):
        if self.write:
            mode = 'r+'
            filters = self.filters
        else:
            mode = 'r'
            filters = None
        files = []
        for f in self.targets:
            files.append(tables.File(f, mode, filters=filters))
        self.files = files
        self.hf = files[0]
        return self


    def __exit__(self, *args):
        for f in self.files:
            f.close()


    def __getitem__(self, key):
        pass


    @property
    def little_h(self):
        try:
            return self._little_h
        except AttributeError:
            self._little_h = self.hf.get_node_attr('/Header', 'little_h')
            return self._little_h


    @property
    def omega_m(self):
        try:
            return self._omega_m
        except AttributeError:
            self._omega_m = self.hf.get_node_attr('/Header', 'omega_m')
            return self._omega_m


    @property
    def sigma_8(self):
        try:
            return self._sigma_8
        except AttributeError:
            self._sigma_8 = self.hf.get_node_attr('/Header', 'sigma_8')
            return self._sigma_8


    @property
    def n_spectral(self):
        try:
            return self._n_spectral
        except AttributeError:
            self._n_spectral = self.hf.get_node_attr('/Header', 'n_spectral')
            return self._n_spectral


    @property
    def box_size(self):
        try:
            return self._box_size
        except AttributeError:
            self._box_size = self.hf.get_node_attr('/Header', 'box_size')
            return self._box_size


    @property
    def mass_unit(self):
        try:
            return self._mass_unit
        except AttributeError:
            self._mass_unit = self.hf.get_node_attr('/Header', 'mass_unit_to_solarmass')*u.solMass
            return self._mass_unit


    @property
    def radius_unit(self):
        try:
            return self._radius_unit
        except AttributeError:
            self._radius_unit = self.hf.get_node_attr('/Header', 'length_unit_to_kpc')*u.kiloparsec
            return self._radius_unit


    @property
    def velocity_unit(self):
        try:
            return self._velocity_unit
        except AttributeError:
            self._velocity_unit = \
                self.hf.get_node_attr('/Header', 'velocity_unit_to_kms')*u.km/u.second
            return self._velocity_unit


    @property
    def m_particle(self):
        try:
            return self._m_particle
        except AttributeError:
            self._m_particle = self.hf.get_node_attr('/Header', 'm_particle')
            return self._m_particle

    @property
    def n_snap(self):
        try:
            return self._n_snap
        except AttributeError:
            self._n_snap = len(self.hf.list_nodes('/')) - 1
            return self._n_snap


    @property
    def redshifts(self):
        try:
            return np.array(self._redshifts)
        except AttributeError:
            self._redshifts = self.hf.get_node('/Header', 'Redshift').read()
            return np.array(self._redshifts)


    def host_id(self, n=1):
        return super().host_id(n)


    def new_id(self, snap, inds):
        return super().new_id(snap, inds)


    def read_cols(self, snap, target, cols=None, rtype='dict', index=None):
        """
        Read data.

        Parameters
        ----------
            target: {'Host', 'Prog', 'Main'}
                Name of the group.
            cols: list
                List of string specifying column names.
            rtype: {'dict', 'pandas', 'records'}
                Type of the return data.
                    dict: python dictionary
                    pandas: pandas data frame
                    records: numpy records
            index: str
                If rtype is 'pandas', specify the index of the data frame.

        Returns
        -------
            Required data.
        """
        n_file = len(self.files)
        if target == 'Prog' and n_file > 1:
            ## No offsets are applied to ProgIndex in host data and Index in
            ## progenitor data. Therefore, it is not meaningfull to read
            ## progenitor data from multiple files.
            raise IOError("Cannot read progentior data from multiple files")
        # Read data
        data_all = [None]*n_file
        offset = [0]*(n_file + 1)
        n_tot = 0
        self._switch_snapshot(snap)
        for i_f, (hf, grp) in enumerate(zip(self.files, self.groups)):
            if target in grp:
                grp = hf.get_node(grp, target)
            else:
                raise IOError(f"Cannot find group '{target}'")
            if i_f == 0:
                if cols is None:
                    cols = [node.name for node in hf.iter_nodes(grp)]
                else:
                    cols = list(cols)
                if index is not None:
                    if index not in cols:
                        cols.append(index)
            data = {}
            for c in cols:
                data[c] = hf.get_node(grp, c).read()
            data_all[i_f] = data
            if len(data) > 0:
                n_tot += len(data[c])
            offset[i_f + 1] = n_tot
        # Merge data
        data_out = {}
        for c in cols:
            offset_inds = 0
            arr = np.zeros(n_tot, dtype=data_all[0][c].dtype)
            for i_f in range(n_file):
                a = data_all[i_f][c]
                if c == 'Index':
                    cond = a < 0
                    a[cond] -= offset_inds
                    offset_inds += np.count_nonzero(cond)
                arr[offset[i_f] : offset[i_f + 1]] = a
            data_out[c] = arr
        # Change data type
        if rtype == 'dict':
            return data_out
        data_out = pd.DataFrame.from_dict(data_out)
        if rtype == 'pandas':
            if index is not None:
                data_out.set_index(index, inplace=True)
            return data_out
        elif rtype == 'records':
            return data_out.to_records(index=False)
        else:
            raise ValueError(f"Unkown rtype: '{rtype}'")


    def read_trees(self,
        mass_key, snap_max, snap_min=0, m_range=None, kwargs_tqdm={}
    ):
        kwargs_tqdm.update(desc='Reading dark forest')
        #
        redshifts = self.redshifts
        # Select host halos
        host_data = self.read_cols(
            snap_max, 'Host', ['ProgIndex', 'ProgNumber', mass_key],
            'pandas', 'Index'
        )
        if m_range is not None:
            m_min, m_max = m_range
            m_host = host_data[mass_key].values
            cond = (m_host >= m_min) & (m_host < m_max)
        else:
            cond = np.full(len(host_data), True)
        n_host = np.count_nonzero(cond)
        #
        for snap in trange(snap_max, snap_min, -1, **kwargs_tqdm):
            level = snap_max - snap + 1
            # Read data
            if snap != snap_max:
                host_data = self.read_cols(
                    snap, 'Host', ['ProgIndex', 'ProgNumber', mass_key],
                    'pandas', 'Index'
                )
            else:
                # Initialise root halos and the pool to read progenitors
                z_root = redshifts[snap_max]
                halo_id_arr = host_data.index[cond]
                m_root_arr = host_data.loc[halo_id_arr, mass_key]
                tree_arr = np.empty(n_host, dtype = object)
                pool = [[] for i_snap in range(snap_max + 1)]
                for i_tree, (halo_id, m_root) \
                    in enumerate(zip(halo_id_arr, m_root_arr)):
                    tree = MergerTree(n_base=10**4)
                    tree.append(m_root, 0)
                    tree_arr[i_tree] = tree
                    pool[snap_max].append((i_tree, 0, halo_id))
            prog_data = self.read_cols(
                snap, 'Prog', ['Index', 'Snap', mass_key], 'records'
            )
            # Trace progenitors
            current_pool = np.asarray(pool[snap])
            if len(current_pool) == 0:
                continue
            prog_pointer = host_data.loc[
                current_pool[:, -1], ['ProgIndex', 'ProgNumber']
            ].values
            cond = prog_pointer[:, -1] > 0
            current_pool = current_pool[cond]
            prog_pointer = prog_pointer[cond]
            for (i_tree, idx_node_host, host_id), (idx, num) \
                in zip(current_pool, prog_pointer):
                tree = tree_arr[i_tree]
                for i_prog, prog in enumerate(prog_data[idx:idx + num]):
                    halo_id = prog['Index']
                    halo_snap = prog['Snap']
                    mass = prog[mass_key]
                    idx_node_new = tree.append(mass, level)
                    if i_prog == 0:
                        tree.link_tail_index(idx_node_host, idx_node_new)
                    else:
                        tree.link_next_index(idx_node_prev, idx_node_new)
                    pool[halo_snap].append((i_tree, idx_node_new, halo_id))
                    idx_node_prev = idx_node_new
            pool[snap] = None
        return tree_arr


    def gather(self,cntr_src, snap, c_target, c_rep=None, is_mass=False, transform=None):
        if c_rep is None:
            c_rep = c_target

        inds = self.read_cols(snap, 'MainA', ['Index'])['Index']
        inds_sim = inds[inds > 0]
        inds_mc = inds[inds < 0]
        df_ret = pd.DataFrame(index=inds)
        grp_name = self._find_group(c_rep, snap)
        if grp_name == 'MainA':
            df_ret[c_target] = self.read_cols(snap, 'MainA', [c_rep])[c_rep]
        else:
            # Combine simulation data
            df_src = pd.DataFrame(
                cntr_src[snap, c_target], index=cntr_src[snap, 'ID'], columns=[c_target]
            )
            df_ret[c_target] = np.zeros(len(df_ret), dtype=df_src.dtypes[c_target])
            df_ret.loc[inds_sim, c_target] = df_src.loc[inds_sim, c_target]
            # Combine extended data
            df_ext = self.read_cols(snap, grp_name, [c_rep], 'pandas', 'Index')
            arr = df_ext.loc[inds_mc, c_rep].values
            if is_mass:
                arr /= self.mass_conversion
            if transform is not None:
                arr = transform(arr)
            arr = np.array(arr, dtype=df_src.dtypes[c_target])
            df_ret.loc[inds_mc, c_target] = arr

        return df_ret


    def remove(self, name, kwargs_tqdm={}):
        kwargs_tqdm.update(desc="Removing '%s'"%name)
        for i_snap in trange(self.n_snap, **kwargs_tqdm):
            for f in self.files:
                try:
                    f.remove_node('/Snapshot_%03d'%i_snap, name, recursive=True)
                except NoSuchNodeError:
                    pass


    def _switch_snapshot(self, snap):
        groups = []
        for f in self.files:
            groups.append(f.get_node('/Snapshot_%03d'%snap))
        self.groups = groups


    def _find_group(self, column, snap):
        self._switch_snapshot(snap)
        for hf, grp in zip(self.files, self.groups):
            if column in hf.get_node(grp, 'Host'):
                grp_name = 'Host'
                break
            elif column in hf.get_node(grp, 'MainA'):
                grp_name = 'MainA'
                break
            elif column in hf.get_node(grp, 'MainB'):
                grp_name = 'MainB'
                break
            else:
                raise ValueError(f"Unstored column '{column}'")
        return grp_name


class DarkForestSaver:
    def __init__(self,
        cntr, fname, mass_key, cols, snap_max, snap_min=0, filters=None, kwargs_tqdm={}
    ):
        self.cntr = cntr
        self.fname = fname
        self.mass_key = np.ravel(mass_key)
        self.cols = np.ravel(cols)
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.filters = filters
        kwargs_tqdm.update(desc='Saving dark dforest')
        self.kwargs_tqdm = kwargs_tqdm


    def run(self):
        #
        cntr = self.cntr
        fname = self.fname
        mass_key = self.mass_key
        cols = self.cols
        snap_max = self.snap_max
        snap_min = self.snap_min
        filters = self.filters
        kwargs_tqdm = self.kwargs_tqdm
        #
        with tables.File(fname, 'w', filters=filters) as hf:
            self._save_header(hf)
            # Trace host halo trees
            n_snap = snap_max + 1
            desc_pool_arr = np.asarray([{} for snap in range(n_snap)])
            for snap in trange(snap_min, n_snap, **kwargs_tqdm):
                # Merge subhalos to their host halo
                desc_pool = desc_pool_arr[snap]
                halo_id_arr = cntr[snap, 'ID']
                host_id_arr = cntr[snap, 'HostHaloID']
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
                self._save_snapshot(hf, desc_pool, snap)
                # Walk forward
                desc_id_arr = cntr[snap, 'Descendant']
                desc_snap_arr = cntr[snap, 'DescendantSnap']
                props_arr = np.asarray([cntr[snap, k] for k in mass_key])
                for desc_id, desc_snap, host_id, halo_id, *props in zip(
                    desc_id_arr, desc_snap_arr, host_id_arr, halo_id_arr, *props_arr
                ):
                    if host_id == -1 and desc_snap <= self.snap_max:
                        target_pool = desc_pool_arr[desc_snap]
                        try:
                            target_pool[desc_id].append((halo_id, snap, *props))
                        except KeyError:
                            target_pool[desc_id] = [(halo_id, snap, *props)]
                desc_pool_arr[snap] = None


    def _save_header(self, hf):
        return save_header(hf, self.cntr)


    def _save_snapshot(self, hf, prog_data, snap):
        #
        cntr = self.cntr
        mass_conversion = cntr.mass_conversion
        mass_key = self.mass_key
        cols = self.cols
        # Save progenitor data
        group_prog = hf.create_group('/Snapshot_%03d'%snap, 'Prog', createparents=True)
        n_prog = len(prog_data)
        plain_data = []
        number = [0]*n_prog
        for i_prog, prog in enumerate(prog_data.values()):
            number[i_prog] = len(prog)
            plain_data.extend(prog)
        df_prog = pd.DataFrame(plain_data, columns=['Index', 'Snap'] + [k for k in mass_key])
        df_prog['Snap'] = df_prog['Snap'].astype('i4')
        df_prog[mass_key] *= mass_conversion
        if len(df_prog) > 0:
            for c in df_prog.columns:
                hf.create_carray(group_prog, c, obj=df_prog[c].values)
        # Save host data
        group_host = hf.create_group('/Snapshot_%03d'%snap, 'Host', createparents=True)
        select_host = cntr[snap, 'HostHaloID'] == cntr.host_id()
        id_ = cntr[snap, 'ID'][select_host]
        df_host = pd.DataFrame(id_, index=id_, columns=['ID'])
        df_host['ProgIndex'] = np.full(len(id_), -1, dtype='i8')
        df_host['ProgNumber'] = np.zeros(len(id_), dtype='i8')
        df_host.loc[prog_data.keys(), 'ProgIndex'] = np.append(0, np.cumsum(number)[:-1])
        df_host.loc[prog_data.keys(), 'ProgNumber'] = number
        hf.create_carray(group_host, 'Index', obj=id_)
        hf.create_carray(group_host, 'ProgIndex', obj=df_host['ProgIndex'].values)
        hf.create_carray(group_host, 'ProgNumber', obj=df_host['ProgNumber'].values)
        for k in mass_key:
            hf.create_carray(group_host, k, obj=cntr[snap, k][select_host]*mass_conversion)
        for c in cols:
            hf.create_carray(group_host, c, obj=cntr[snap, c][select_host])


def save_header(hf, cntr):
    header = hf.create_group('/', 'Header')
    # Save redshifts
    hf.create_carray(header, 'Redshift', obj=cntr.redshifts)
    # Save attributes
    hf.set_node_attr(header, 'little_h', cntr.little_h)
    hf.set_node_attr(header, 'omega_m', cntr.omega_m)
    hf.set_node_attr(header, 'sigma_8', cntr.sigma_8)
    hf.set_node_attr(header, 'n_spectral', cntr.n_spectral)
    hf.set_node_attr(header, 'box_size', cntr.box_size)
    hf.set_node_attr(header, 'mass_unit_to_solarmass', cntr.mass_unit.value)
    hf.set_node_attr(header, 'length_unit_to_kpc', cntr.radius_unit.value)
    hf.set_node_attr(header, 'velocity_unit_to_kms', cntr.velocity_unit.value)
    hf.set_node_attr(header, 'm_particle', cntr.m_particle)
