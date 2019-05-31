from .halotree import *
from .logger import CreateLogger

import numpy as np


logger = CreateLogger(__name__)


def Prune(m_root, t_root, m_prog, t_prog, z_prog, m_cut):
    """
    This function has three purposes:
        a) Linearly interpolate the most massive progenitor with time, and
           sort the mass in descending order.
        b) Find `z_target' and remove progentiros which are not at 'z_target'.
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


def ScatterHostData(host_data, seed, comm, master_rank):
    if comm is None:
        return host_data, seed
    else:
        i_rank = comm.Get_rank()
        n_rank = comm.Get_size()
        if i_rank == master_rank:
            host_data.sort_values('Number', inplace = True)
        for i_send in range(n_rank):
            if i_send == master_rank:
                host_data_retval = host_data[i_send::n_rank]
                seed_retval = seed[i_send::n_rank]
                continue
            if i_rank == master_rank:
                comm.send(host_data[i_send::n_rank], dest = i_send, tag = 0)
                comm.send(seed[i_send::n_rank], dest = i_send, tag = 1)
            elif i_rank == i_send:
                host_data_retval = comm.recv(source = master_rank, tag = 0)
                seed_retval = comm.recv(source = master_rank, tag = 1)
    return host_data_retval, seed_retval
