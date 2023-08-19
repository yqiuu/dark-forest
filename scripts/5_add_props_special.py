import os
import sys
import logging

#sys.path.legend should point to where the dark-forest is located
sys.path.append('/home/bsreedha/models/dark-forest/')

import yaml
import h5py
import tables
import numpy as np
import dforest
from tqdm import trange
from dforest import DensityField, GridTool
from local_module import SpecialConstructor

import hmf

import astropy.units as u
kms_to_MpcMyr = u.km.to(u.Mpc)/u.s.to(u.Myr)

def resample_by_factor(array, sample_fac=4):
    arr_shape = array.shape[0]
    arr = array.reshape(arr_shape//sample_fac, sample_fac,
                    arr_shape//sample_fac, sample_fac,
                    arr_shape//sample_fac, sample_fac)
    arr = arr.mean(axis=-1).mean(axis=3).mean(axis=1)
    return arr

class DynamicDensityField(DensityField):
    def move_halos(self, x_h, y_h, z_h, inds, m1, m2, snap1, return_disps=False):
        ret = super().move_halos(x_h, y_h, z_h, inds, m1, m2, snap1, return_disps)
        if snap1 > 108:
            n_grid_v = 256
        elif snap1 > 90:
            n_grid_v = 128
        else:
            n_grid_v = 64
        self.gt_vel = GridTool(n_grid_v, self.gt_vel.box_size)
        return ret

    def read_density_field(self, snap):
        path_grid = "/fred/oz009/GENESIS/L210_N4320_Grids_Halos_Trees-balu/grids"
        with h5py.File(os.path.join(path_grid, 'snap_%04d.hdf5'%snap), 'r') as hf:
            den_dm = np.ravel(resample_by_factor(np.asarray(hf['PartType1/Grids/Density']), 2))
        return den_dm/np.mean(den_dm) - 1
        
    def read_transform_field(self, snap):
        path_grid = "/fred/oz009/GENESIS/L210_N4320_Grids_Halos_Trees-balu/grids"
        fname = "snap_%04d.hdf5"%snap
        #fname = 'Snapshot_%03d_Vel_%d.hdf5'%(snap, self.gt_vel.n_grid)
        with h5py.File(os.path.join(path_grid, fname), 'r') as hf:
            vx_dm = np.ravel(resample_by_factor(self.redshifts[snap]*np.asarray(hf['PartType1/Grids/Vx']), 4))*kms_to_MpcMyr
            vy_dm = np.ravel(resample_by_factor(self.redshifts[snap]*np.asarray(hf['PartType1/Grids/Vy']), 4))*kms_to_MpcMyr
            vz_dm = np.ravel(resample_by_factor(self.redshifts[snap]*np.asarray(hf['PartType1/Grids/Vz']), 4))*kms_to_MpcMyr
        factor = ((1 + self.redshifts[snap])**2)*self.D[snap]/self.dDdt[snap]
        return vx_dm*factor, vy_dm*factor, vz_dm*factor


dforest.logger.setLevel("INFO")
inputs = yaml.safe_load(open(sys.argv[1]))
fname_source = inputs['file_source']
path_ext = inputs['path_out']
prefix_ext = inputs['prefix_out']
mass_key = inputs['mass_key']
snap_max = inputs['snap_max']
snap_min = inputs['snap_min']
i_part = int(sys.argv[2])
#
filters = tables.Filters(9, 'blosc:lz4')

with dforest.DarkForestConnector(path_ext, prefix_ext, write=i_part, filters=filters) as cntr_ext, \
        dforest.VELOCIraptorConnector(fname_source) as cntr_src:
    kwargs_tqdm = {'disable': False}
    cntr_ext.remove('MainA', kwargs_tqdm=kwargs_tqdm)
    cntr_ext.remove('MainB', kwargs_tqdm=kwargs_tqdm)
    dforest.ForwardConstructor(
        cntr_ext, cntr_src, snap_max, snap_min, kwargs_tqdm=kwargs_tqdm
    ).run()
    seed = 2092*10 + i_part
    rstate = np.random.RandomState(seed=seed)
    cols = {
        'spin':'lambda_B', 'L_x':'Lx', 'L_y':'Ly', 'L_z':'Lz',
        'm_main': mass_key, 'm_vir': mass_key, 'r_vir':'R_200crit', 'v_vir':'V_200crit',
        'x': 'Xc', 'y': 'Yc', 'z': 'Zc',
    }
    SpecialConstructor(
        cntr_ext, cntr_src, snap_max, snap_min, n_min=100, kwargs_tqdm=kwargs_tqdm
    ).run()
    # Position
    n_grid_p = 512 
    n_grid_v = 256
    box_size = cntr_src.box_size
    redshifts = cntr_src.redshifts
    ages = cntr_src.ages
    emf = cntr_src.emf()
    emf.update(hmf_model='PS')
    r_max = box_size
    density_field = DynamicDensityField(
        emf, redshifts, ages, n_grid_p, n_grid_v, box_size, r_max=r_max
    )
    dforest.PositionConstructor(
        cntr_ext, cntr_src, cols, density_field, 20000, .1, snap_max, snap_min,
        comoving=False, kwargs_tqdm=kwargs_tqdm
    ).run()
