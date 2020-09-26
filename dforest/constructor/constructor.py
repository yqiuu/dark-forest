import os
import gc
from abc import ABC, abstractmethod 

from ..logger import logger, dividing_line, memory_usage

import numpy as np
import pandas as pd
from tables import NoSuchNodeError
from tqdm import trange


__all__ = ['Constructor']


class Constructor(ABC):
    """Abstract class to add a new halo property.

    Overwrite ``_add_property`` method to enable the class. This method should
    return a ``pandas.DataFrame`` object and the data will be saved in the
    file specified by ``cntr_ext``. There are two groups to save the data,
    i.e. 'MainA' and 'MainB'. The former contains all halos (including
    subhalos) that are not exculded from the origal trees (specified by
    ``cntr_src``) and all MC halos, while the latter only has MC halos. If
    the new columns overwrites the original halo properties, save them in
    `MainA`; otherwise they should be saved in 'MainB'. The code will check
    the consistency of the data using the indices of the DataFrame. The
    indices in 'MainA' and 'MainB' are pre-determined by
    ``ForwardConstructor``.

    Parameters
    ----------
    cntr_ext : Connector instance
        Connector to the extended trees.
    cntr_src : Connector instance
        Connector to the original trees.
    col_inh : list
        Column names of inherited data.
    col_req : list
        Column names of request data.
    snap_max : int
        Maximum snapshot
    snap_min : int
        Minimum snapshot
    read_prog : list
        Column names of progenitor data. Use empty list if no properties is
        needed.
    inherit : bool
        If True, save data in 'MainB', else save data in 'MainA'.
    postfix : str
        Add '_{postfix}' to the column names of the data to save.
    message : str
        Message to describe the progress bar.
    kwargs_tqdm : dict
        Keyword arguments for tqdm.
    """
    def __init__(self,
        col_inh, col_req, snap_max, snap_min,
        read_prog, inherit, forward, postfix, message, kwargs_tqdm
    ):
        self.col_inh = list(col_inh)
        self.col_req = list(col_req)
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.read_prog = read_prog
        self.inherit = inherit
        self.forward = forward
        self.postfix = postfix
        self.message = message
        self.kwargs_tqdm = kwargs_tqdm


    def run(self, cntr_ext, cntr_src):
        """Run the constructor."""
        self.cntr_ext = cntr_ext
        self.cntr_src = cntr_src
        self._set_column_names(self.col_req, self.snap_min)
        redshifts = self.cntr_ext.redshifts
        n_snap = self.snap_max - self.snap_min + 1
        for i_snap, snap in enumerate(self._range()):
            logger.info(f"{self.message}: {i_snap+1}/{n_snap} ({memory_usage()})")
            z = redshifts[snap]
            df_req = self._prepare_req_data(snap)
            df_inh = self._prepare_inh_data(snap)
            df_inh.index = df_inh.index.astype(df_req.index.dtype)
            list_prog = self._prepare_prog_data(snap)
            df_new = self._add_property(df_inh, df_req, list_prog, snap, z)
            self._save_output(df_new, snap)
            gc.collect()
            logger.info(dividing_line())


    def _set_column_names(self, col_req, snap_min):
        node_host = [
            node.name for node in self.cntr_ext.hf.iter_nodes('/Snapshot_%03d/Host'%snap_min)
        ]
        col_host = []
        col_main = []
        for c in col_req:
            if c in node_host:
                col_host.append(c)
            else:
                col_main.append(c)
        self.col_host = col_host
        self.col_main = col_main


    def _range(self):
        kwargs_tqdm = self.kwargs_tqdm
        kwargs_tqdm.update(desc=self.message)
        if self.forward:
            return trange(self.snap_min, self.snap_max + 1, **kwargs_tqdm)
        else:
            return trange(self.snap_max, self.snap_min - 1, -1, **kwargs_tqdm)


    def _prepare_prog_data(self, snap):
        cntr_ext = self.cntr_ext
        read_prog = self.read_prog
        if len(read_prog) > 0:
            try:
                rec_prog = cntr_ext.read_cols(snap, 'Prog', read_prog, 'records')
            except OSError:
                return None
            if len(read_prog) == 1:
                list_prog = list(rec_prog[read_prog[0]])
            else:
                list_prog = rec_prog
            return list_prog
        else:
            return None


    def _prepare_req_data(self, snap):
        df_host = self.cntr_ext.read_cols(snap, 'Host', self.col_host, 'pandas', 'Index')
        if len(self.col_main) > 0:
            df_main = self.cntr_ext.read_cols(snap, 'MainA', self.col_main, 'pandas', 'Index')
            df_req = pd.concat([df_host, df_main], axis=1, join='inner')
        else:
            df_req = df_host
        return df_req


    def _prepare_inh_data(self, snap):
        cntr_ext = self.cntr_ext
        cntr_src = self.cntr_src
        if cntr_ext == cntr_src:
            return pd.DataFrame()
        else:
            halo_id_sim = cntr_src[snap, 'ID']
            df_inh = pd.DataFrame(index=halo_id_sim)
            for col in self.col_inh:
                try:
                    df_inh[col] = cntr_src[snap, col]
                except KeyError:
                    pass
            try:
                halo_id_fix \
                    = cntr_ext.read_cols(snap, 'MainA', ['Index'])['Index']
            except IOError:
                raise Exception(
                    "Run FowardRebuilder before adding any other properties"
                ) from None
            df_inh = df_inh.loc[halo_id_fix[halo_id_fix > 0]]
            if len(df_inh) > 0:
                return df_inh
            else:
                return pd.DataFrame()


    @abstractmethod
    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        """
        Parameters
        ----------
        df_inh: pandas.DataFrame:
            Halo data from the original trees.
            Columns:
                Specified by `col_inh`.
            Indices:
                All halos (including subhalos) that are not exclude from the
                original trees.
        df_req: pandas.DataFrame
            Halo data read from the extended trees.
            Columns:
                Specified by `col_req`.
            Indices:
                All MC halos.
        list_prog: list
            Progenitor data from the extended trees
        snap: int
            Snapshot number to read the data.
        z: float
            Redshift of the given snapshot.

        Returns
        -------
        pandas.DataFrame
            The indices of the DataFrame should match those in `MainA` or
            'MainB'.
        """
        return pd.DataFrame()


    def _save_output(self, df_new, snap):
        hf = self.cntr_ext.hf
        postfix = '' if self.postfix is None else f'_{self.postfix}'
        name = 'MainB' if self.inherit else 'MainA'
        group = hf.get_node(f'/Snapshot_{snap:03d}', name)
        try:
            inds = self.cntr_ext.read_cols(snap, name, ['Index'])['Index']
        except NoSuchNodeError:
            inds = None
        if inds is None or np.array_equal(inds, df_new.index):
            for col in df_new.columns:
                col_save = f'{col}{postfix}'
                try:
                    hf.remove_node(group, col_save)
                except NoSuchNodeError:
                    pass
                hf.create_carray(group, col_save, obj=df_new[col].values)
        else:
            raise ValueError("Indices mismatch.")

