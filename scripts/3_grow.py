"""Grow one forest partition with MPI; arguments are YAML config and partition index."""

import os
import sys
import logging

#sys.path.append should point to where the dark-forest is located locally
# sys.path.append("/home/yqiu/repository/dark-forest/")
sys.path.append('/home/bsreedha/models/dark-forest/')

import yaml
import numpy as np
import pandas as pd
import dforest
import tables
from mpi4py import MPI

#
logger = logging.getLogger('dforest')
logger.setLevel('INFO')
# Set MPI comm
comm = MPI.COMM_WORLD
# Set parts
i_part = int(sys.argv[2])
n_part = len(os.listdir("outputs/parts"))
# Set inputs
if comm.Get_rank() == 0:
    print('i_part: %d'%i_part)
    print(''.join(open(sys.argv[1]).readlines()), '\n')

inputs = yaml.safe_load(open(sys.argv[1]))
file_in = inputs['file_in']
path_out = inputs['path_out']
prefix_out = inputs['prefix_out']
mass_key = inputs['mass_key']
snap_max = inputs['snap_max']
snap_min = inputs['snap_min']
m_res = inputs['m_res']
params_mc = inputs['params_mc']
params_ag = inputs['params_ag']
#
filters = tables.Filters(9, 'blosc:lz4')

with dforest.DarkForestConnector(file_in) as cntr:
    h = cntr.little_h
    redshifts = cntr.redshifts
    m_p = cntr.m_particle
    volume = cntr.volume*h**3
    #
    params_ag['m_min_cut'] *= m_p
    params_ag['m_max_cut'] *= m_p
    grow_kit = dforest.GrowKit(m_res=m_res, **params_mc)
    # Sample MC halos
    seed = 2092*100 + i_part
    rstate = np.random.RandomState(seed=seed)
    #m_hmf, dndm = np.load("hmf_L210N1536_z5.npy")
    #m_hmf, dndm = np.load("hmf_L105N2048_z5.npy")
    m_hmf, dndm = np.load("hmf_L35L105_z5_combined.npy")
    root_halos = dforest.sample_mass_func(
        dndm, m_hmf, m_res, params_ag['m_min_cut'], volume/n_part, rstate
    )
    #
    inds_sub = np.load("outputs/parts/L210_part%d.npy"%i_part)
    #
    dforest.grow_halos(
        cntr=cntr,
        fname="%s/%s_part%d.hdf5"%(path_out, prefix_out, i_part),
        mass_key=mass_key,
        grow_kit=grow_kit,
        snap_max=snap_max,
        snap_min=snap_min,
        params_ag=params_ag,
        root_halos=root_halos,
        inds_sub=inds_sub,
        filters=filters,
        seed=seed,
        comm=comm
    )
