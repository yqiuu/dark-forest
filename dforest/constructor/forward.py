import os
import gc

from ..logger import logger, dividing_line, memory_usage
from .constructor import Constructor

import numpy as np
import pandas as pd
import tables
from tqdm import trange
from tables import NoSuchNodeError


__all__ = [
    'ForwardRebuilder',
    'ForwardConstructor'
]


def generate_file_name(cntr_ext, i_part=None):
    prefix = cntr_ext.prefix
    if prefix is None:
        fname = os.path.join(os.path.split(cntr_ext.targets[0])[0], '_tmp_forward_fix.hdf5')
    elif i_part is None:
        fname = cntr_ext.targets[0].replace(prefix, f'_{prefix}_forward_fix')
    else:
        fname = cntr_ext.file_names[i_part].replace(prefix, f'_{prefix}_forward_fix')
    return fname


class ForwardRebuilder:
    def __init__(self, cntr_ext, cntr_src, snap_max, snap_min=0, kwargs_tqdm={}):
        self.cntr_ext = cntr_ext
        self.cntr_src = cntr_src
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.kwargs_tqdm = kwargs_tqdm


    def run(self):
        with tables.File(generate_file_name(self.cntr_ext), 'w') as hf:
            snap_jump = self._rebuild(hf)
            hf.set_node_attr('/', 'snap_jump', snap_jump)


    def _prepare_source_data(self, snap):
        cntr_src = self.cntr_src
        df_source = pd.DataFrame(index=cntr_src[snap, 'ID'])
        df_source['HostHaloID'] = cntr_src[snap, 'HostHaloID']
        df_source['DescendantSnap'] = cntr_src[snap, 'DescendantSnap']
        df_source['Descendant'] = cntr_src[snap, 'Descendant']
        df_source['ForestID'] = cntr_src[snap, 'ForestID']
        return df_source


    def _put_top_halos(self, pool_sub, df_source, snap):
        halo_id_surv = self.cntr_ext.read_cols(snap, 'Host', ['Index'])['Index']
        halo_id_surv = halo_id_surv[halo_id_surv > 0]

        halo_id = df_source.index.values
        host_id = df_source['HostHaloID'].values.copy()
        select_host = host_id == self.cntr_src.host_id()
        host_id[select_host] = halo_id[select_host]
        select_top = np.in1d(halo_id, pool_sub.keys(), invert=True)
        for h_id, surv in zip(halo_id[select_top], np.in1d(host_id[select_top], halo_id_surv)):
            pool_sub[h_id] = surv


    def _set_survive_inds(self, pool_sub, df_source):
        df_source['Survive'] = 0
        df_source.loc[list(pool_sub.keys()), 'Survive'] = list(pool_sub.values())


    def _cut_descendants(self, df_source, snap, snap_max):
        # [1] Assuem that it points to itself when there is no descendant halo
        select_cut = df_source['DescendantSnap'] > snap_max
        df_source.loc[select_cut, 'DescendantSnap'] = snap
        df_source.loc[select_cut, 'Descendant'] = df_source.index[select_cut] # [1]


    def _walk_forward(self, pool, df_source):
        for desc_snap, desc_id, surv in \
            df_source[['DescendantSnap', 'Descendant', 'Survive']].values:
            pool[desc_snap][desc_id] = surv


    def _fix_subhalo_descendant(self, pool, df_source):
        df_sub = df_source[df_source['HostHaloID'] != self.cntr_src.host_id()]
        if len(df_sub > 0):
            halo_id = df_sub.index.values
            desc_snap = df_sub['DescendantSnap'].values
            desc_id = df_sub['Descendant'].values
            halo_id_sub = [
                s_id for s_id, desc_snap, desc_id in zip(halo_id, desc_snap, desc_id) \
                    if not pool[desc_snap][desc_id]
            ]
            host_id = df_source.loc[halo_id_sub, 'HostHaloID'].values
            n_redirect = len(host_id)
            if n_redirect > 0:
                logger.info(f"Redirecting {n_redirect} subhalos")
                df_source.loc[halo_id_sub, 'DescendantSnap'] \
                    = df_source.loc[host_id, 'DescendantSnap'].values
                df_source.loc[halo_id_sub, 'Descendant'] \
                    = df_source.loc[host_id, 'Descendant'].values


    def _save_output(self, hf, df_source, snap):
        df_save = df_source[df_source['Survive']]
        grp = hf.create_group('/', f'Snapshot_{snap:03d}')
        n_halo = len(np.union1d(
            self.cntr_ext.read_cols(snap, 'Host', ['Index'])['Index'], df_save.index.values
        ))
        hf.set_node_attr(grp, 'n_halo', n_halo)
        if len(df_save) > 0:
            hf.create_carray(
                grp, 'Index', obj=df_save.index.values.astype('i8')
            )
            hf.create_carray(
                grp, 'HostHaloID', obj=df_save['HostHaloID'].values.astype('i8')
            )
            hf.create_carray(
                grp, 'DescendantSnap', obj=df_save['DescendantSnap'].values.astype('i4')
            )
            hf.create_carray(
                grp, 'Descendant', obj=df_save['Descendant'].values.astype('i8')
            )
            hf.create_carray(
                grp, 'ForestID', obj=df_save['ForestID'].values.astype('i8')
            )


    def _rebuild(self, hf):
        # Load local variables
        cntr_ext = self.cntr_ext
        cntr_src = self.cntr_src
        snap_max = self.snap_max
        snap_min = self.snap_min
        kwargs_tqdm = self.kwargs_tqdm
        #
        n_snap = snap_max + 1
        df_source_all = [None]*n_snap
        pool = [{} for i_snap in range(n_snap)]
        snap_target = np.arange(n_snap)
        kwargs_tqdm.update(desc="Rebuilding original trees")
        for snap in trange(snap_min, n_snap, **kwargs_tqdm):
            logger.info(f"Processing snapshot {snap}/{n_snap} ({memory_usage()})")
            pool_sub = pool[snap]
            df_source = self._prepare_source_data(snap)
            self._put_top_halos(pool_sub, df_source, snap)
            self._set_survive_inds(pool_sub, df_source)
            self._cut_descendants(df_source, snap, snap_max)
            self._walk_forward(pool, df_source)
            if len(df_source) > 0:
                snap_target[snap] = max(df_source['DescendantSnap'])
            else:
                snap_target[snap] = snap_min - 1
            df_source_all[snap] = df_source
            for snap_t in range(snap_min, snap):
                snap_target_it = snap_target[snap_t]
                if snap == snap_target_it:
                    df_fix = df_source_all[snap_t]
                    self._fix_subhalo_descendant(pool, df_fix)
                    self._save_output(hf, df_fix, snap_t)
                    # Clear memory
                    if snap_t == snap_min or snap_target_it > max(snap_target[snap_min:snap_t]):
                        for snap_c in range(snap_min, snap_t + 1):
                            df_source_all[snap_c] = None
                            pool[snap_c] = None
            if snap == snap_max:
                self._save_output(hf, df_source, snap)
            gc.collect()
            logger.info(dividing_line())
        snap_jump = max(snap_t - snap for snap_t, snap in zip(snap_target, range(n_snap)))
        return snap_jump


class ForwardConstructor(Constructor):
    def __init__(self, cntr_ext, cntr_src, snap_max, snap_min=0, kwargs_tqdm={}):
        super().__init__(
            cntr_ext=cntr_ext,
            cntr_src=cntr_src,
            col_inh=[],
            col_req=[],
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=['Index'],
            inherit=False,
            forward=False,
            postfix=None,
            message='Adding forward pointers',
            kwargs_tqdm=kwargs_tqdm
        )
        self.kwargs_tqdm = kwargs_tqdm
        self.pool_sim = [{} for snap in range(snap_max + 1)]
        self.df_forward_next = None
        self.forest_id_max = cntr_src.max_forest_id(snap_max, snap_min)
        #
        self.test_mode = False


    def run(self):
        with tables.File(generate_file_name(self.cntr_ext), 'r') as hf_fix:
            self.snap_jump = hf_fix.get_node_attr('/', 'snap_jump')
            self.hf_fix = hf_fix
            super().run()


    def _get_offset(self, snap):
        cntr_ext = self.cntr_ext
        offset = 0
        for i_part in range(cntr_ext.i_part):
            with tables.File(generate_file_name(cntr_ext, i_part)) as hf:
                try:
                    offset += hf.get_node_attr(f'/Snapshot_{snap:03d}', 'n_halo')
                except NoSuchNodeError:
                    pass
        return offset


    def _prepare_req_data(self, snap):
        return self.cntr_ext.read_cols(
            snap, 'Host', ['ProgIndex', 'ProgNumber'], 'pandas', 'Index'
        )


    def _prepare_inh_data(self, snap):
        try:
            grp = self.hf_fix.get_node(f'/Snapshot_{snap:03d}')
        except NoSuchNodeError:
            grp = None
        df_inh = pd.DataFrame()
        if grp is not None and 'Index' in grp:
            for col in ['Index', 'HostHaloID', 'DescendantSnap', 'Descendant', 'ForestID']:
                df_inh[col] = self.hf_fix.get_node(grp, col).read()
            df_inh.set_index('Index', inplace=True)
        return df_inh

    # def _prepare_inh_data(self, snap):
    #     grp = self.hf_fix.get_node(f'/Snapshot_{snap:03d}')
    #     df_inh = pd.DataFrame()
    #     if 'Index' in grp:
    #         for col in ['Index', 'HostHaloID', 'DescendantSnap', 'Descendant', 'ForestID']:
    #             df_inh[col] = self.hf_fix.get_node(grp, col).read()
    #         df_inh.set_index('Index', inplace=True)
    #     return df_inh


    def _generate_pointer_mc(self, df_inh, df_req, list_prog, snap, offset):
        # Load local variables
        cntr_src = self.cntr_src
        snap_max = self.snap_max
        forest_id_max = self.forest_id_max
        df_forward_next = self.df_forward_next
        #
        select_mc_halo = df_req.index < 0
        df_forward_mc = pd.DataFrame(index=df_req.index[select_mc_halo])
        n_halo_mc = len(df_forward_mc)
        n_halo_sim = len(df_inh)
        df_forward_mc['ID'] = cntr_src.new_id(
            snap, np.arange(offset + n_halo_sim, offset + n_halo_sim + n_halo_mc)
        )
        df_forward_mc['HostHaloID'] = cntr_src.host_id(n_halo_mc)
        if snap == snap_max:
            df_forward_mc['DescendantSnap'] = snap
            df_forward_mc['Descendant'] = df_forward_mc['ID'] # [1]
            df_forward_mc['ForestID'] = np.arange(forest_id_max, forest_id_max + n_halo_mc)
        else:
            df_forward_mc = df_forward_mc.join(df_forward_next)
        # Put DescendantSnap/Descendant/ForestID into the pool
        progenitors = {}
        halo_id_host_sim = df_req.index[~select_mc_halo]
        if n_halo_sim > 0:
            df_forward = df_inh.loc[df_req.index[~select_mc_halo]]
            df_forward = df_forward.append(df_forward_mc)
            df_forward = df_forward.loc[df_req.index]
        else:
            df_forward = df_forward_mc
        cond = df_req['ProgNumber'] > 0
        for (idx, num), (h_id, f_id) in zip(
            df_req.loc[cond, ['ProgIndex', 'ProgNumber']].values,
            df_forward.loc[cond, ['ID', 'ForestID']].values
        ):
            for p_id in list_prog[idx:idx + num]:
                if p_id < 0:
                    progenitors[p_id] = (snap, h_id, f_id)
        self.df_forward_next = pd.DataFrame.from_dict(
            progenitors, orient='index', columns=['DescendantSnap', 'Descendant', 'ForestID']
        )
        return df_forward_mc


    def _generate_pointer_ns(self, df_inh, snap, offset):
        """
        Generate new IDs for simulation halos, and change Descendant/HostHaloID
        accordingly.
        """
        # Load local variables
        cntr_src = self.cntr_src
        pool_sim = self.pool_sim
        snap_max = self.snap_max
        snap_jump = self.snap_jump
        #
        df_inh.insert(0, 'ID', cntr_src.new_id(snap, np.arange(offset, offset + len(df_inh))))
        #
        pool_sim[snap] = {old_id: new_id for old_id, new_id in zip(df_inh.index, df_inh['ID'])}
        # Fix host halo IDs
        select_sub_halo = df_inh['HostHaloID'] != cntr_src.host_id()
        halo_id_sub = df_inh.loc[df_inh.loc[select_sub_halo, 'HostHaloID'], 'ID'].values
        df_inh.loc[select_sub_halo, 'HostHaloID'] = halo_id_sub
        # Generate new Descendant for simulations halos
        desc_snap = df_inh['DescendantSnap'].values
        desc_id = df_inh['Descendant'].values
        df_inh['Descendant'] = list(map(
            lambda d_snap, d_id: pool_sim[d_snap][d_id], desc_snap, desc_id
        ))
        # Clean memory
        snap_clean = snap + snap_jump
        if snap_clean <= snap_max:
            pool_sim[snap_clean] = None


    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        offset = self._get_offset(snap)
        if len(df_inh) > 0:
            self._generate_pointer_ns(df_inh, snap, offset)
        df_forward = self._generate_pointer_mc(df_inh, df_req, list_prog, snap, offset)
        df_forward = df_inh.append(df_forward)
        return df_forward


    def _save_output(self, df_new, snap):
        hf = self.cntr_ext.hf
        grp_snap = hf.get_node(f'/Snapshot_{snap:03d}')
        if 'MainA' in grp_snap:
            hf.remove_node(grp_snap, 'MainA', recursive=True)
        if 'MainB' in grp_snap:
            hf.remove_node(grp_snap, 'MainB', recursive=True)
        grp_a = hf.create_group(f'/Snapshot_{snap:03d}', 'MainA')
        for col in df_new.columns:
            hf.create_carray(grp_a, col, obj=df_new[col].values)
        inds = df_new.index.values.astype('i8')
        hf.create_carray(grp_a, 'Index', obj=inds)
        grp_b = hf.create_group(f'/Snapshot_{snap:03d}', 'MainB')
        hf.create_carray(grp_b, 'Index', obj=inds[inds < 0])
