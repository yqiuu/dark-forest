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
            host_data = pd.read_hdf(store, 'Host', columns = ['ProgIndex', 'ProgNumber', mass_key])
        return host_data, prog_data
    else:
        return None, None


def ScatterData(
    host_data_raw, prog_data_raw, next_host_data, pool_data,
    mass_key, m_cut, filters, comm, master_rank
):
    """
    Scatter the raw data to different cores.

    Notes:
        This function applies the mass cut to host halo data.
        If a host halo has no progenitors, the coresponding progenitor data
        is `None`.

    Parameters
    ----------
        host_data_raw: pandas.DataFrame
            Raw host halo data read from the disk.
        prog_data_raw: pandas.DataFrame
            Raw progenitor data read from the disk.
        next_host_data: numpy structured array
            Extra host halo data. These can be from manually added halos or new
            halos generated in a previous snapshot.
        pool_data: list
            Indices to select host halos.
        mass_key: string
            The name of the mass key.
        m_cut: float
            The mass cut applied to the host halos.
        filters: dict
            Functions to select host halos at the first snapshot.
        comm: MPI.Comm
            An MPI communicator. Use `None` when MPI is not used.
        master_rank: int
            The rank of the master core.
    """

    if comm is not None:
        i_rank = comm.Get_rank()
        n_rank = comm.Get_size()
    else:
        i_rank = master_rank
    if i_rank == master_rank:
        if pool_data is None:
            host_data = host_data_raw[host_data_raw[mass_key] > m_cut]
            host_data = host_data.to_records()
            cond = np.full(len(host_data), True)
            for key, item in filters.items():
                cond &= item(host_data[key])
            host_data = host_data[cond]
        else:
            host_data = host_data_raw.loc[pool_data]
            host_data = host_data.to_records()
        host_data = np.asarray(
            host_data[list(next_host_data.dtype.names)], dtype = next_host_data.dtype
        )
        host_data = np.append(host_data, next_host_data, axis = 0)
        host_data[::-1].sort(order = ['ProgNumber', mass_key])
        #
        prog_data_raw = list(prog_data_raw.to_records())
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
    host_chunk, new_prog_chunk, next_host_chunk, pool_chunk, stats_chunk, comm, master_rank
):
    if comm is None:
        i_rank = master_rank
        host_data = host_chunk
        new_prog_data = new_prog_chunk
        next_host_data = next_host_chunk
        pool_data = pool_chunk
        stats_data = stats_chunk
    else:
        i_rank = comm.Get_rank()
        # Gather data
        host_data = comm.gather(host_chunk, root = master_rank)
        new_prog_list = comm.gather(new_prog_chunk, root = master_rank)
        next_host_data = comm.gather(next_host_chunk, root = master_rank)
        pool_data = comm.gather(pool_chunk, root = master_rank)
        stats_data = comm.gather(stats_chunk, root = master_rank)
        if i_rank == master_rank:
            host_data = np.concatenate(host_data)
            new_prog_data = []
            for item in new_prog_list:
                new_prog_data.extend(item)
            new_prog_data = np.asarray(new_prog_data, dtype = object)
            next_host_data = np.concatenate(next_host_data)
            pool_data = np.concatenate(pool_data)
            stats_data = np.concatenate(stats_data)
    if i_rank == master_rank:
        cond = host_data['ProgNumber'] > 0
        new_prog_data = new_prog_data[cond]
        host_data['ProgIndex'][cond] = np.append(0, np.cumsum(host_data['ProgNumber'][cond])[:-1])
        host_data['ProgIndex'][~cond] = -1
        return host_data, new_prog_data, next_host_data, pool_data, stats_data
    else:
        return None, None, None, None, None


def SaveData(path, prefix, host_data, prog_data, stats_data, snap, comm, master_rank):
    if comm is None or comm.Get_rank() == master_rank:
        prog_data = np.concatenate(prog_data)
        df_prog = pd.DataFrame(prog_data)
        df_prog.set_index('ID', inplace = True)
        df_host = pd.DataFrame(host_data)
        df_host.set_index('ID', inplace = True)
        insert_column = 0
        for key in stats_data.dtype.names:
            df_host.insert(insert_column, key, stats_data[key])
            insert_column += 1
        with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap)), "w") as store:
            store.append('Prog', df_prog)
            store.append('Host', df_host)


def NewID(n_halo, snap, offset):
    new_id = 1 + snap*10**12 + offset + np.arange(n_halo)
    return -new_id


def InterploateMass(m_prog, s_prog, m_root, t_root, t_target, ages):
    """
    Interploate the mass of the most massive progenitor with time. (inplace)
    """
    if len(m_prog) > 0:
        ind_primary = np.argmax(m_prog)
        m_primary = m_prog[ind_primary]
        t_primary = ages[s_prog[ind_primary]]
        m_intp = m_root + (m_primary - m_root)/(t_primary - t_root)*(t_target - t_root)
        m_prog[ind_primary] = m_intp


def Augment(
    halo_kit, m_root, z_root, m_prog, z_target, m_res, params_ag, params_mc, rstate, n_base
):
    """
    Given a pruned branch, generate Monte Carlo tree to find a match.
    """
    # Load algorithm parameters
    m_min_cut = params_ag['m_min_cut']
    m_max_cut = params_ag['m_max_cut']
    eps_cut = params_ag['eps_cut']
    n_trail_cut = params_ag['n_trail_cut']

    eps_match = eps_match_0 = params_ag['eps_match']
    n_trail_match = params_ag['n_trail_match']

    n_max_trail = params_ag['n_max_trail']
    # Initialise loop variables
    m_cut = m_min_cut
    next_level = 1
    initial_fit = -1.
    best_fit = initial_fit

    n_cut = 0
    n_match = 0
    n_trail = 0
    # Sort progenitors by mass
    m_prog[::-1].sort()
    #
    while True:
        # Relax matching condition
        use_best = False
        if n_cut == n_trail_cut:
            n_cut = 0
            n_match = 0
            best_fit = initial_fit
            if len(m_prog) > 0:
                m_cut *= (1 + eps_cut)
                m_cut = min(m_cut, m_max_cut, max(m_prog))
            else:
                m_cut = m_max_cut
            # Apply mass cut
            m_prog = m_prog[m_prog >= m_cut]
        elif n_match == n_trail_match:
            n_cut = 0
            n_match = 0
            eps_match *= (1 + eps_match_0)
            if best_fit != initial_fit:
                use_best = True
        # Create a new realisation of a simple branch
        if n_trail == n_max_trail:
            m_prog_new = np.array([], dtype = 'f8')
            break
        if use_best:
            m_prog_trail = m_best
            n_prog_trail = n_best
        else:
            tree = Tree_Parkinson08(
                halo_kit, m_root, z_root, z_target, m_res,
                rstate = rstate, n_base = n_base, **params_mc
            )
            m_prog_trail = tree.Halos(next_level)['mass']
            n_prog_trail = len(m_prog_trail)
            n_trail += 1
        # Check matching condition
        n_prog = len(m_prog)
        if n_prog_trail >= n_prog:
            fit_arr = abs(m_prog_trail[:n_prog]/m_prog - 1.)
            if all(fit_arr < eps_match):
                m_prog_new = m_prog_trail[n_prog:]
                break
            else:
                # Save the best fit
                fit = np.sum(fit_arr);
                if fit < best_fit or best_fit == initial_fit:
                    best_fit = fit
                    m_best = m_prog_trail
                    n_best = n_prog_trail
                n_match += 1
        else:
            n_cut += 1
    m_prog_new = m_prog_new[m_prog_new < m_cut]
    return m_prog_new, m_cut, eps_match, n_trail


def AugmentChunk(
    host_chunk, prog_chunk, pool_list_chunk, mass_key, snap,
    z_root, t_root, z_target, t_target, redshifts, ages,
    halo_kit, m_cut, m_res, params_ag, params_mc, rstate, n_base, comm,
):
    dtype_host = host_chunk.dtype
    dtype_prog = np.dtype([('ID', 'i8'), ('Snap', 'i4'), (mass_key, 'f8')])
    dtype_stats = np.dtype([('MassCutFix', 'f4'), ('EpsMatchFix', 'f4'), ('Trail', 'i4')])
    # Make empty arrays for outputs
    n_host_chunk = len(host_chunk)
    new_prog_chunk = np.empty(n_host_chunk, dtype = object)
    next_host_chunk = np.empty(n_host_chunk, dtype = object)
    stats_chunk = np.zeros(n_host_chunk, dtype = dtype_stats)
    #
    n_new_halo = 0
    for i_chunk in range(n_host_chunk):
        host, prog = host_chunk[i_chunk], prog_chunk[i_chunk]
        m_root = host[mass_key]
        if prog is None:
            prog = np.array([], dtype = dtype_prog)
        else:
            prog = np.asarray(prog, dtype = dtype_prog)
            # Apply mass cut to progenitor data
            prog = prog[prog[mass_key] > m_cut]
            # Fix host data
            host['ProgNumber'] = len(prog)
        m_prog = np.copy(prog[mass_key])
        InterploateMass(m_prog, prog['Snap'], m_root, t_root, t_target, ages)
        m_unres, m_cut_fix, eps_match_fix, n_trail = Augment(
            halo_kit, m_root, z_root, m_prog, z_target,
            m_res, params_ag, params_mc, rstate, n_base
        )
        stats_chunk[i_chunk] = m_cut_fix, eps_match_fix, n_trail
        # Create new progenitor and next host data
        n_unres = len(m_unres)
        if n_unres > 0:
            # Generate new halo IDs
            new_id = NewID(n_unres, snap - 1, n_new_halo)
            n_new_halo += n_unres
            host['ProgNumber'] += n_unres
            # Process new progenitor data
            prog = np.array(prog[list(dtype_prog.names)], dtype = dtype_prog)
            prog_unres = np.full(n_unres, -1, dtype = dtype_prog)
            prog_unres['ID'] = new_id
            prog_unres['Snap'] = np.full(n_unres, snap - 1, dtype = 'i4')
            prog_unres[mass_key] = m_unres
            prog = np.append(prog, prog_unres)
            new_prog_chunk[i_chunk] = prog
            # Process next host data
            next_host = np.full(n_unres, -1, dtype = dtype_host)
            next_host['ID'] = new_id
            next_host['ProgNumber'] = 0
            next_host[mass_key] = m_unres
            next_host_chunk[i_chunk] = next_host
        else:
            new_prog_chunk[i_chunk] = prog
            next_host_chunk[i_chunk] = np.array([], dtype = dtype_host)
        # Add progenitor IDs to the pool
        for prog_id, prog_snap, _ in prog:
            if prog_id > 0:
                pool_list_chunk[prog_snap].append(prog_id)
    next_host_chunk = np.concatenate(next_host_chunk)
    # Fix offset in new IDs
    if comm is not None:
        i_rank = comm.Get_rank()
        offset = comm.allgather(n_new_halo)
        offset = np.append(0, np.cumsum(offset))[i_rank]
        for new_prog in new_prog_chunk:
            new_prog['ID'][new_prog['ID'] < 0] -= offset
        next_host_chunk['ID'] -= offset
    return new_prog_chunk, next_host_chunk, stats_chunk


def HybridGrowth(
    hc_path, hc_prefix, out_path, out_prefix, snap_min, snap_max, mass_key,
    redshifts, ages, halo_kit, m_res, params_ag, params_mc, new_halos = None,
    filters = {}, seed = 209204, n_base = 5000, comm = None, master_rank = 0
):
    rstate = CreateRandomState(seed, comm)
    m_cut = params_ag['m_min_cut']
    # Initialise new halos
    dtype_host = [('ID', 'i8'), ('ProgIndex', 'i4'), ('ProgNumber', 'i4'), (mass_key, 'f8')]
    if new_halos is None:
        next_host_data = np.array([], dtype = dtype_host)
    else:
        n_new = len(new_halos)
        next_host_data = np.zeros(n_new, dtype = dtype_host)
        next_host_data['ID'] = NewID(n_new, snap_max, 0)
        next_host_data['ProgIndex'] = np.full(n_new, -1, dtype = 'i4')
        next_host_data[mass_key] = new_halos
    #
    pool_data, pool_list_chunk = None, [[] for i in range(snap_max + 1)]
    #
    for snap in range(snap_max, snap_min - 1, -1):
        host_data_raw, prog_data_raw = ReadHaloData(
            hc_path, hc_prefix, snap, mass_key, comm, master_rank
        )
        host_chunk, prog_chunk = ScatterData(
            host_data_raw, prog_data_raw, next_host_data, pool_data,
            mass_key, m_cut, filters, comm, master_rank
        )
        z_root, t_root = redshifts[snap], ages[snap]
        z_target, t_target = redshifts[snap - 1], ages[snap - 1]
        new_prog_chunk, next_host_chunk, stats_chunk = AugmentChunk(
            host_chunk, prog_chunk, pool_list_chunk, mass_key, snap,
            z_root, t_root, z_target, t_target, redshifts, ages,
            halo_kit, m_cut, m_res, params_ag, params_mc, rstate, n_base, comm
        )
        host_data, prog_data, next_host_data, pool_data, stats_data = GatherData(
            host_chunk, new_prog_chunk, next_host_chunk,
            pool_list_chunk[snap - 1], stats_chunk, comm, master_rank
        )
        SaveData(
            out_path, out_prefix, host_data, prog_data, stats_data, snap, comm, master_rank
        )

