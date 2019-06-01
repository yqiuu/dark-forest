from .halotree import *
from .logger import CreateLogger

import os

import numpy as np
import pandas as pd

logger = CreateLogger(__name__)


def Prune(m_root, t_root, m_prog, t_prog, z_prog, m_cut):
    """
    This function has three purposes:
        a) Linearly interpolate the most massive progenitor with time, and
           sort the mass in descending order.
        b) Find `z_target' and remove progenitors which are not at 'z_target'.
        c) Remove progenitors whose mass is below `m_cut`.
    Mass and time should be in linear space, and their units are irrelevant.    
    """

    ind_m_max = np.argmax(m_prog)
    m2 = m_prog[ind_m_max]
    t2 = t_prog[ind_m_max]
    ind_z_min = np.argmin(z_prog)
    z_target = z_prog[ind_z_min]
    t_intp = t_prog[ind_z_min]
    m_intp = m_root + (m2 - m_root)/(t2 - t_root)*(t_intp - t_root)
    m_prog[ind_m_max] = m_intp
    z_prog[ind_m_max] = z_target
    #
    m_prog = m_prog[z_prog == z_target]
    m_prog = m_prog[m_prog > m_cut]
    m_prog[::-1].sort()

    return m_prog, z_target


def Augment(
    halo_kit, m_root, z_root, m_prog, z_target, m_cut, m_res,
    eps, n_trail, n_max, params_mc, rstate, n_base
):
    """
    Given a pruned branch, generate Monte Carlo tree to find a match.
    """
    n_prog = len(m_prog)
    initial_fit = -1.
    best_fit = initial_fit

    i_trail = 0
    i_max = 0
    while True:
        if i_max == n_max:
            if best_fit == initial_fit:
                logger.warning("cannot find a match and return an empty array")
                m_prog_new = np.array([])
                break
            else:
                logger.warning("cannot find a match and return the best match")
                m_prog_new = m_best[n_prog:]
                break
        use_best = False
        if i_trail == n_trail:
            i_trail = 0
            eps *= (1 + eps)
            if best_fit != initial_fit:
                use_best = True
            i_max += 1
        i_trail += 1                   
        if use_best:
            m_prog_trail = m_best
            n_prog_trail = n_best
        else:
            tree = Tree_Parkinson08(
                halo_kit, m_root, z_root, z_target, m_res,
                rstate = rstate, n_base = n_base, **params_mc
            )
            m_prog_trail = tree.Halos(1)['mass']
            n_prog_trail = len(m_prog_trail)
        if n_prog_trail >= n_prog:
            fit_arr = np.abs(m_prog_trail[:n_prog]/m_prog - 1.)
            if np.all(fit_arr < eps):
                m_prog_new = m_prog_trail[n_prog:]
                break
            else:
                fit = np.sum(fit_arr);
                if fit < best_fit or best_fit == initial_fit:
                    best_fit = fit
                    m_best = m_prog_trail
                    n_best = n_prog_trail
    return m_prog_new[m_prog_new <= m_cut], i_max*n_trail + i_trail


def CreateRandomState(seed, comm):
    if comm is None:
        return RandomStateGSL(seed)
    else:
        return RandomStateGSL(seed + comm.Get_rank())


def ReadHaloData(hc_path, hc_prefix, snap, mass_key, m_cut, comm, master_rank):
    if comm is None or comm.Get_rank() == master_rank:
        host_data = pd.read_hdf(os.path.join(hc_path, '%s_host_%d.hdf5'%(hc_prefix, snap)))
        host_data = host_data.loc[
            host_data[mass_key] > m_cut, ('IndexStart', 'IndexEnd', 'Number', mass_key)
        ]
        host_data.sort_values('Number', ascending = False, inplace = True)
        host_data = host_data.to_records()[['IndexStart', 'IndexEnd', mass_key]]
        prog_data = pd.read_hdf(os.path.join(hc_path, '%s_prog_%d.hdf5'%(hc_prefix, snap)))
        prog_data = prog_data.to_records()[['Snap', mass_key]]
        return host_data, prog_data
    else:
        return None, None


def GetChunkSize(host_chunk, comm, master_rank):
    if comm is None:
        n_host = host_chunk
        return n_host, n_host
    else:
        n_host = len(host_chunk)
        n_chunk = max(comm.allgather(n_host))
        return n_chunk, n_host


def ScatterHostData(host_data, comm, master_rank):
    if comm is None:
        return host_data
    else:
        i_rank = comm.Get_rank()
        n_rank = comm.Get_size()
        if i_rank == master_rank:
            data = []
            for i_scatter in range(n_rank):
                data.append(host_data[i_scatter::n_rank])
        else:
            data = None
        host_chunk = comm.scatter(data, root = master_rank)
        return host_chunk


def ScatterProgData(prog_data, index_start, index_end, comm, master_rank):
    if comm is None:
        return prog_data[index_start:index_end]
    else:
        i_rank = comm.Get_rank()
        n_rank = comm.Get_size()
        index_start_arr = comm.gather(index_start, root = master_rank)
        index_end_arr = comm.gather(index_end, root = master_rank)
        if i_rank == master_rank:
            data = []
            for idx_s, idx_e in zip(index_start_arr, index_end_arr):
                data.append(prog_data[idx_s:idx_e])
        else:
            data = None
        progenitors = comm.scatter(data, root = master_rank)
        return progenitors


def HybridGrowth(
    handler, hc_path, hc_prefix, snap_min, snap_max, mass_key, m_cut, m_res,
    params_ag, params_mc, seed = 209204, comm = None, master_rank = 0
):
    rstate = CreateRandomState(seed, comm)
    for snap in range(snap_max, snap_min - 1, -1):
        host_data, prog_data = ReadHaloData(
            hc_path, hc_prefix, snap, mass_key, m_cut, comm, master_rank
        )
        host_chunk = ScatterHostData(host_data, comm, master_rank)
        n_chunk, n_host = GetChunkSize(host_chunk, comm, master_rank)
        for i_chunk in range(n_chunk):
            if i_chunk < n_host:
                index_start, index_end, m_root = host_chunk[i_chunk]
            else:
                index_start, index_end, m_root = 0, 1, None
            progenitors = ScatterProgData(prog_data, index_start, index_end, comm, master_rank)
            print(comm.Get_rank(), progenitors, m_root)



