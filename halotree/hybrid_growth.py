from .halotree import *
from .logger import CreateLogger

import os

import numpy as np
import pandas as pd

logger = CreateLogger(__name__)


def CreateRandomState(seed, comm):
    if comm is None:
        return RandomStateGSL(seed)
    else:
        return RandomStateGSL(seed + comm.Get_rank())


def ReadHaloData(path, prefix, snap, mass_key, comm, master_rank):
    if comm is None or comm.Get_rank() == master_rank:
        logger.info("Read snapshot %d"%snap)
        with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap))) as store:
            prog_data = pd.read_hdf(store, 'Prog', columns = ['Snap', mass_key])
            prog_data = prog_data.to_records()
            host_data = pd.read_hdf(store, 'Host', columns = ['ProgIndex', 'ProgNumber', mass_key])
            host_data = host_data.to_records()
        return host_data, prog_data
    else:
        return None, None


def ScatterData(host_data_raw, prog_data_raw, next_host_data, mass_key, m_cut, comm, master_rank):
    if comm is not None:
        i_rank = comm.Get_rank()
        n_rank = comm.Get_size()
    else:
        i_rank = master_rank
    if i_rank == master_rank:
        host_data = host_data_raw[host_data_raw[mass_key] > m_cut]
        host_data = np.asarray(
            host_data[list(next_host_data.dtype.names)], dtype = next_host_data.dtype
        )
        host_data = np.append(host_data, next_host_data, axis = 0)
        host_data[::-1].sort(order = ['ProgNumber', mass_key])
        prog_data_raw = list(prog_data_raw)
        prog_data = np.full(len(host_data), None)
        cond = host_data['ProgNumber'] > 0
        idx_arr = host_data['ProgIndex'][cond]
        num_arr = host_data['ProgNumber'][cond]
        prog_data[np.where(cond)] = list(
            map(lambda idx, num:prog_data_raw[idx:idx + num], idx_arr, num_arr)
        )
        logger.info("Process %d host halos"%len(host_data))
    if comm is None:
        return host_data, prog_data
    else:
        if i_rank == master_rank:
            host_sub = []
            prog_sub = []
            for i_scatter in range(n_rank):
                host_sub.append(host_data[i_scatter::n_rank])
                prog_sub.append(prog_data[i_scatter::n_rank])
        else:
            host_sub = None
            prog_sub = None
        host_chunk = comm.scatter(host_sub, root = master_rank)
        prog_chunk = comm.scatter(prog_sub, root = master_rank)
        return host_chunk, prog_chunk


def GatherData(
    host_chunk, new_prog_chunk, next_host_chunk, n_trail_chunk, comm, master_rank
):
    if comm is None:
        i_rank = master_rank
        host_data = host_chunk
        new_prog_data = new_prog_chunk
        next_host_data = next_host_chunk
        n_trail_data = n_trail_chunk
    else:
        i_rank = comm.Get_rank()
        # Gather data
        host_data = comm.gather(host_chunk, root = master_rank)
        new_prog_list = comm.gather(new_prog_chunk, root = master_rank)
        next_host_data = comm.gather(next_host_chunk, root = master_rank)
        n_trail_data = comm.gather(n_trail_chunk, root = master_rank)
        if i_rank == master_rank:
            host_data = np.concatenate(host_data)
            new_prog_data = []
            for item in new_prog_list:
                new_prog_data.extend(item)
            new_prog_data = np.asarray(new_prog_data, dtype = object)
            next_host_data = np.concatenate(next_host_data)
            n_trail_data = np.concatenate(n_trail_data)
    if i_rank == master_rank:
        cond = host_data['ProgNumber'] > 0
        new_prog_data = new_prog_data[cond]
        host_data['ProgIndex'][cond] = np.append(0, np.cumsum(host_data['ProgNumber'][cond])[:-1])
        return host_data, new_prog_data, next_host_data, n_trail_data
    else:
        return None, None, None, None


def SaveData(path, prefix, host_data, prog_data, n_trail_data, snap, comm, master_rank):
    if comm is None or comm.Get_rank() == master_rank:
        prog_data = np.concatenate(prog_data)
        df_prog = pd.DataFrame(prog_data)
        df_prog.set_index('ID', inplace = True)
        df_host = pd.DataFrame(host_data)
        df_host.set_index('ID', inplace = True)
        df_host.insert(2, 'Trail', n_trail_data)
        with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap)), "w") as store:
            store.append('Prog', df_prog)
            store.append('Host', df_host)


def Prune(host, prog, mass_key, t_root, z_target, t_target, redshifts, ages, m_cut):
    # Apply mass cut to progenitors
    m_prog = prog[mass_key]
    cond = m_prog > m_cut
    prog = prog[cond]
    m_prog = m_prog[cond]
    # Change the number of progentiors (inplace)
    n_prog = len(prog)
    host['ProgNumber'] = n_prog
    if n_prog > 0:
        # Set some local variables
        snap = prog['Snap']
        z_prog = redshifts[snap]
        t_prog = ages[snap]
        m_root = host[mass_key]
        # Intepolate the mass of the most massive object with time
        ind_m_max = np.argmax(m_prog)
        m2 = m_prog[ind_m_max]
        t2 = t_prog[ind_m_max]
        m_intp = m_root + (m2 - m_root)/(t2 - t_root)*(t_target - t_root)
        m_prog[ind_m_max] = m_intp
        z_prog[ind_m_max] = z_target
        # Select all progenitors at the target redshift
        m_prog = m_prog[z_prog == z_target]
        # Sort progenitors by mass
        m_prog[::-1].sort()
    return prog, m_prog


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
                logger.warning("Cannot find a match and return an empty array")
                m_prog_new = np.array([])
                break
            else:
                logger.warning("Cannot find a match and return the best match")
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


def AugmentChunk(
    host_chunk, prog_chunk, mass_key, snap, z_root, t_root, z_target, t_target, redshifts, ages,
    halo_kit, m_cut, m_res, params_ag, params_mc, rstate, n_base, comm,
):
    dtype_host = host_chunk.dtype
    dtype_prog = np.dtype([('ID', 'i8'), ('Snap', 'i4'), (mass_key, 'f8')])
    # Make empty arrays for outputs
    n_host_chunk = len(host_chunk)
    new_prog_chunk = np.empty(n_host_chunk, dtype = object)
    next_host_chunk = np.empty(n_host_chunk, dtype = object)
    n_trail_chunk = np.zeros(n_host_chunk, dtype = 'i4')
    #
    n_new_halo = 0
    for i_chunk in range(n_host_chunk):
        host, prog = host_chunk[i_chunk], prog_chunk[i_chunk]
        m_root = host[mass_key]
        if prog is None:
            prog = np.array([], dtype = dtype_prog)
            m_pruned = np.array([], dtype = 'f8')
        else:
            prog = np.asarray(prog, dtype = dtype_prog)
            prog, m_pruned = Prune(
                host, prog, mass_key, t_root, z_target, t_target, redshifts, ages, m_cut
            )
        m_unres, n_trail_chunk[i_chunk] = Augment(
            halo_kit, m_root, z_root, m_pruned, z_target, m_cut, m_res,
            params_mc = params_mc, rstate = rstate, n_base = n_base, **params_ag
        )
        n_unres = len(m_unres)
        if n_unres > 0:
            # Generate new halo ID
            new_id = 1 + (snap - 1)*10**12 + n_new_halo + np.arange(n_unres)
            new_id = -new_id
            n_new_halo += n_unres
            host['ProgNumber'] += n_unres
            # Process new progenitor data
            prog = np.array(prog[list(dtype_prog.names)], dtype = dtype_prog)
            prog_unres = np.full(n_unres, -1, dtype = dtype_prog)
            prog_unres['ID'] = new_id
            prog_unres['Snap'] = np.full(n_unres, snap - 1, dtype = 'i4')
            prog_unres[mass_key] = m_unres
            new_prog_chunk[i_chunk] = np.append(prog, prog_unres)
            # Process next host data
            next_host = np.full(n_unres, -1, dtype = dtype_host)
            next_host['ID'] = new_id
            next_host['ProgNumber'] = 0
            next_host[mass_key] = m_unres
            next_host_chunk[i_chunk] = next_host
        else:
            new_prog_chunk[i_chunk] = prog
            next_host_chunk[i_chunk] = np.array([], dtype = dtype_host)
    next_host_chunk = np.concatenate(next_host_chunk)
    # Fix offset in new IDs
    if comm is not None:
        i_rank = comm.Get_rank()
        offset = comm.allgather(n_new_halo)
        offset = np.append(0, np.cumsum(offset))[i_rank]
        for new_prog in new_prog_chunk:
            new_prog['ID'][new_prog['ID'] < 0] -= offset
        next_host_chunk['ID'] -= offset
    return new_prog_chunk, next_host_chunk, n_trail_chunk


def HybridGrowth(
    hc_path, hc_prefix, out_path, out_prefix, snap_min, snap_max, mass_key,
    redshifts, ages, halo_kit, m_cut, m_res, params_ag, params_mc,
    seed = 209204, n_base = 5000, comm = None, master_rank = 0
):
    rstate = CreateRandomState(seed, comm)
    dtype_host = [('ID', 'i8'), ('ProgIndex', 'i4'), ('ProgNumber', 'i4'), (mass_key, 'f8')]
    next_host_data = np.array([], dtype = dtype_host)
    for snap in range(snap_max, snap_min - 1, -1):
        host_data_raw, prog_data_raw = ReadHaloData(
            hc_path, hc_prefix, snap, mass_key, comm, master_rank
        )
        host_chunk, prog_chunk = ScatterData(
            host_data_raw, prog_data_raw, next_host_data, mass_key, m_cut, comm, master_rank
        )
        z_root, t_root = redshifts[snap], ages[snap]
        z_target, t_target = redshifts[snap - 1], ages[snap - 1]
        new_prog_chunk, next_host_chunk, n_trail_chunk = AugmentChunk(
            host_chunk, prog_chunk, mass_key, snap, z_root, t_root, z_target, t_target,
            redshifts, ages, halo_kit, m_cut, m_res, params_ag, params_mc, rstate, n_base, comm
        )
        host_data, prog_data, next_host_data, n_trail_data = GatherData(
            host_chunk, new_prog_chunk, next_host_chunk, n_trail_chunk, comm, master_rank
        )
        SaveData(
            out_path, out_prefix, host_data, prog_data, n_trail_data, snap, comm, master_rank
        )

