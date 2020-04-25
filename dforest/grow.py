from ._dtree import *
from .connector.dforest import save_header
from .logger import logger, dividing_line, memory_usage

import os

import tables
import numpy as np
import pandas as pd
from tables import NoSuchNodeError
from astropy import units


__all__ = [
    'grow_halos',
    'create_partitions',
    'DarkSeed'
]


def create_output(cntr, fname, filters, comm, master_rank):
    if comm is None or comm.Get_rank() == master_rank:
        if isinstance(cntr, DarkSeed):
            with tables.File(fname, 'w', filters=filters) as hf:
                header = hf.create_group('/', 'Header')
                # Save redshifts
                hf.create_carray(header, 'Redshift', obj=cntr.redshifts)
                # Save attributes
                emf = cntr.emf
                cosmo = emf.cosmo
                hf.set_node_attr(header, 'little_h', cosmo.H0.value*0.01)
                hf.set_node_attr(header, 'omega_m', cosmo.Om0)
                hf.set_node_attr(header, 'sigma_8', emf.sigma_8)
                hf.set_node_attr(header, 'n_spectral', emf.n)
                hf.set_node_attr(header, 'box_size', cntr.box_size)
                hf.set_node_attr(header, 'mass_unit_to_solarmass', 1e-10)
                hf.set_node_attr(header, 'length_unit_to_kpc', 1.)
                hf.set_node_attr(header, 'velocity_unit_to_kms', 1.)
                hf.set_node_attr(header, 'm_particle', 0.)
        else:
            with tables.File(fname, 'w', filters=filters) as hf:
                save_header(hf, cntr)


def init_root_halos(cntr, mass_key, snap_max, root_halos):
    dtype_host = [
        ('Index', 'i8'), ('ProgIndex', 'i4'), ('ProgNumber', 'i4'),
        ('LeafType', 'i4'), (mass_key, 'f8')
    ]
    if root_halos is None:
        next_host_data = np.array([], dtype=dtype_host)
    else:
        n_new = len(root_halos)
        next_host_data = np.zeros(n_new, dtype=dtype_host)
        next_host_data['Index'] = create_new_id(n_new, snap_max, 0) #[1]
        next_host_data['ProgIndex'] = np.full(n_new, -1, dtype='i4')
        next_host_data['LeafType'] = np.full(n_new, 2, dtype='i4')
        next_host_data[mass_key] = root_halos
    return next_host_data


def read_data(cntr, mass_key, snap, snap_max, inds_sub, comm, master_rank):
    if comm is None or comm.Get_rank() == master_rank:
        if isinstance(cntr, DarkSeed):
            return None, None
        else:
            try:
                prog_data = cntr.read_cols(snap, 'Prog', ['Snap', mass_key], 'pandas', 'Index')
                logger.info("Read in %d progenitor halos"%len(prog_data))
            except NoSuchNodeError:
                prog_data = None
            try:
                host_data = cntr.read_cols(
                    snap, 'Host', ['ProgIndex', 'ProgNumber', mass_key], 'pandas', 'Index'
                )
                logger.info("Read in %d host halos"%len(host_data))
            except NoSuchNodeError:
                host_data = None
            if snap == snap_max and inds_sub is not None:
                host_data = host_data.iloc[inds_sub]
            return host_data, prog_data
    else:
        return None, None


def scatter_data(
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
        if host_data_raw is None:
            host_data = np.array([], dtype=next_host_data.dtype)
        else:
            if pool_data is None:
                host_data = host_data_raw[host_data_raw[mass_key] > m_cut]
                host_data.index.name = 'Index'
                host_data.insert(2, 'LeafType', np.full(len(host_data), 0, dtype='i4'))
                host_data = host_data.to_records()
                cond = np.full(len(host_data), True)
                for key, item in filters.items():
                    cond &= item(host_data[key])
                host_data = host_data[cond]
            else:
                host_data = host_data_raw.loc[pool_data]
                host_data.insert(2, 'LeafType', np.full(len(host_data), 0, dtype='i4'))
                host_data.index.name = 'Index'
                host_data = host_data.to_records()
        host_data = np.asarray(
            host_data[list(next_host_data.dtype.names)], dtype = next_host_data.dtype
        )
        host_data = np.append(host_data, next_host_data, axis = 0)
        host_data[::-1].sort(order = ['ProgNumber', mass_key])
        #
        prog_data = np.full(len(host_data), None)
        if prog_data_raw is not None:
            prog_data_raw = list(prog_data_raw.to_records())
            cond = host_data['ProgNumber'] > 0
            idx_arr = host_data['ProgIndex'][cond]
            num_arr = host_data['ProgNumber'][cond]
            prog_data[np.where(cond)] = list(
                map(lambda idx, num:prog_data_raw[idx:idx + num], idx_arr, num_arr)
            )
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


def gather_data(
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
            new_prog_data = np.asarray(new_prog_data)
            next_host_data = np.concatenate(next_host_data)
            pool_data = np.concatenate(pool_data)
            stats_data = np.concatenate(stats_data)
    if i_rank == master_rank:
        cond = host_data['ProgNumber'] > 0
        new_prog_data = new_prog_data[cond]
        if len(new_prog_data) > 0:
            new_prog_data = np.concatenate(new_prog_data)
        host_data['ProgIndex'][cond] = np.append(0, np.cumsum(host_data['ProgNumber'][cond])[:-1])
        host_data['ProgIndex'][~cond] = -1
        return host_data, new_prog_data, next_host_data, pool_data, stats_data
    else:
        return None, None, None, None, None


def save_data(
    cntr, fname, host_data, prog_data, stats_data, snap, snap_min, filters, comm, master_rank
):
    if comm is None or comm.Get_rank() == master_rank:
        with tables.File(fname, 'a', filters=filters) as hf:
            # Save progenitor data
            group_prog = hf.create_group('/Snapshot_%03d'%snap, 'Prog', createparents=True)
            n_prog_data = len(prog_data)
            if n_prog_data > 0:
                logger.info("Saving data of %d progenitor halos"%n_prog_data)
                for c in prog_data.dtype.names:
                    hf.create_carray(group_prog, c, obj=prog_data[c])
            # Save host data for the last snapshot
            if snap - 1 == snap_min:
                group_last = hf.create_group('/Snapshot_%03d'%snap_min, 'Host', createparents=True)
                if n_prog_data > 0:
                    logger.info("Saving data for the last snapshot")
                    prog_index = np.full(n_prog_data, -1, dtype=host_data.dtype['ProgIndex'])
                    hf.create_carray(group_last, 'ProgIndex', obj=prog_index)
                    prog_number = np.full(n_prog_data, 0, dtype=host_data.dtype['ProgNumber'])
                    hf.create_carray(group_last, 'ProgNumber', obj=prog_number)
                    leaf_type = get_leaf_type(host_data, prog_data)
                    hf.create_carray(group_last, 'LeafType', obj=leaf_type)
                    for c in prog_data.dtype.names:
                        if c != 'Snap':
                            hf.create_carray(group_last, c, obj=prog_data[c])
            # Save host data
            group_host = hf.create_group('/Snapshot_%03d'%snap, 'Host', createparents=True)
            n_host_data = len(host_data)
            if n_host_data > 0:
                logger.info("Saving data of %d host halos"%n_host_data)
                for c in host_data.dtype.names:
                    hf.create_carray(group_host, c, obj=host_data[c])
                if not isinstance(cntr, DarkSeed):
                    for c in stats_data.dtype.names:
                        hf.create_carray(group_host, c, obj=stats_data[c])


def get_leaf_type(host_data, prog_data):
    cond = host_data['ProgNumber'] > 0
    halo_id = host_data['Index'][cond]
    prog_index = host_data['ProgIndex'][cond]
    prog_number = host_data['ProgNumber'][cond]
    leaf_type_host = host_data['LeafType'][cond]
    leaf_type_prog = np.zeros(len(prog_data), dtype=host_data.dtype['LeafType'])
    prog_id = prog_data['Index']
    for h_id, idx, num, lt_host in zip(halo_id, prog_index, prog_number, leaf_type_host):
        if h_id > 0:
            lt_prog = (prog_id[idx: idx + num] < 0).astype(host_data.dtype['LeafType'])
            leaf_type_prog[idx : idx+num] = lt_prog
        else:
            leaf_type_prog[idx : idx+num] = lt_host
    return leaf_type_prog


def create_new_id(n_halo, snap, offset):
    new_id = 1 + snap*10**12 + offset + np.arange(n_halo)
    return -new_id


def interpolate_mass(m_prog, s_prog, m_root, t_root, t_target, ages):
    """
    Interploate the mass of the most massive progenitor with time. (inplace)
    """
    if len(m_prog) > 0:
        ind_primary = np.argmax(m_prog)
        m_primary = m_prog[ind_primary]
        t_primary = ages[s_prog[ind_primary]]
        m_intp = m_root + (m_primary - m_root)/(t_primary - t_root)*(t_target - t_root)
        m_prog[ind_primary] = m_intp


def augment_simple_branch(
    planter, host, prog, mass_key, m_root, z_root, z_target, params_ag,
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
    prog[::-1].sort(order = mass_key)
    prog_0 = prog.copy()
    n_prog_0 = len(prog_0)
    m_prog = prog[mass_key]
    #
    while True:
        # Relax matching condition
        use_best = False
        if n_cut == n_trail_cut:
            n_cut = 0
            best_fit = initial_fit
            if n_prog_0 > 0:
                m_cut *= (1 + eps_cut)
                m_cut = min(m_cut, m_max_cut, max(m_prog))
            else:
                m_cut = m_max_cut
            # Apply mass cut
            prog = prog[prog[mass_key] >= m_cut]
            host['ProgNumber'] = len(prog)
            m_prog = prog[mass_key]
        if n_match == n_trail_match:
            n_match = 0
            eps_match *= (1 + eps_match_0)
            if best_fit != initial_fit:
                use_best = True
        if n_trail == n_max_trail:
            m_prog_new = np.array([], dtype='f8')
            prog = prog_0[prog_0[mass_key] >= m_min_cut]
            host['ProgNumber'] = len(prog)
            break
        # Create a new realisation of a simple branch
        if use_best:
            m_prog_trail = m_best
            n_prog_trail = n_best
        else:
            tree = planter.grow(m_root, z_root, z_target)
            m_prog_trail = tree.leaf(next_level)
            n_prog_trail = len(m_prog_trail)
            n_trail += 1
        # Check matching condition
        n_prog = len(m_prog)
        m_prog_new = m_prog_trail[n_prog:]
        if n_prog_trail >= n_prog:
            fit_arr = abs(m_prog_trail[:n_prog]/m_prog - 1.)
            if all(fit_arr < eps_match):
                if all(m_prog_new < m_cut):
                    break
                else:
                    n_cut += 1
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
    return prog, m_prog_new, m_cut, eps_match, n_trail


def augment_chunk(
    planter, host_chunk, prog_chunk, pool_list_chunk,
    mass_key, snap, z_root, t_root, z_target, t_target,
    redshifts, ages, m_cut, params_ag, comm,
):
    #
    dtype_host = host_chunk.dtype
    dtype_prog = np.dtype([('Index', 'i8'), ('Snap', 'i4'), (mass_key, 'f8')])
    dtype_stats = np.dtype([('MassCutFix', 'f4'), ('EpsMatchFix', 'f4'), ('Trail', 'i4')])
    # Make empty arrays for outputs
    n_host_chunk = len(host_chunk)
    if n_host_chunk == 0:
        new_prog_chunk = np.array([], dtype=dtype_prog)
        next_host_chunk = np.array([[], []], dtype=dtype_host)
        stats_chunk = np.array([], dtype=dtype_stats)
    else:
        new_prog_chunk = np.empty(n_host_chunk, dtype=object)
        next_host_chunk = np.empty(n_host_chunk, dtype=object)
        stats_chunk = np.zeros(n_host_chunk, dtype=dtype_stats)
    #
    i_rank = 0 if comm is None else comm.Get_rank()
    logger.info("Growing %d halos (i_rank = %d)"%(n_host_chunk, i_rank))
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
        interpolate_mass(m_prog, prog['Snap'], m_root, t_root, t_target, ages)
        prog, m_unres, m_cut_fix, eps_match_fix, n_trail \
            = augment_simple_branch(
                planter, host, prog, mass_key,
                m_root, z_root, z_target, params_ag,
            )
        stats_chunk[i_chunk] = m_cut_fix, eps_match_fix, n_trail
        # Create new progenitor and next host data
        n_unres = len(m_unres)
        if n_unres > 0:
            # Generate new halo IDs
            new_id = create_new_id(n_unres, snap - 1, n_new_halo)
            n_new_halo += n_unres
            host['ProgNumber'] += n_unres
            # Process new progenitor data
            prog = np.array(prog[list(dtype_prog.names)], dtype = dtype_prog)
            prog_unres = np.full(n_unres, -1, dtype = dtype_prog)
            prog_unres['Index'] = new_id
            prog_unres['Snap'] = np.full(n_unres, snap - 1, dtype = 'i4')
            prog_unres[mass_key] = m_unres
            prog = np.append(prog, prog_unres)
            new_prog_chunk[i_chunk] = prog
            # Process next host data
            next_host = np.full(n_unres, -1, dtype = dtype_host)
            next_host['Index'] = new_id
            next_host['ProgNumber'] = 0
            next_host['LeafType'] = 2 if host['LeafType'] == 2 else 1
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
            new_prog['Index'][new_prog['Index'] < 0] -= offset
        next_host_chunk['Index'] -= offset
    return new_prog_chunk, next_host_chunk, stats_chunk


def grow_halos(
    cntr, planter, fname, mass_key, snap_max, snap_min=0, params_ag=None,
    root_halos=None, inds_sub=None, filters=None, comm=None, master_rank=0
):
    """Primary function to grow halos.

    This function can be used to extend halo merger trees of N-body
    simulations or to grow a sample Monte Carlo merger trees.

    Parameters
    ----------
    cntr : Connector or DarkSeed instance
        Object specifies the cosmology and the timesteps. It also gives access
        to the N-body simulation data if applicable.
    fname : str
        Path of the output file.
    mass_key : str
        Label for the halo mass.
    grow_kit : GrowKit instance
        Object specifies the parameters to grow a single Monte Carlo halo
        trees.
    snap_max: int
        Maximum snapshot.
    snap_min: int, optional
        Minimum snapshot.
    params_ag : dict
        Parameters of the augmentation algorithm.
    root_halos : 1-D array
        Masses of additional halos to be augmented to the N-body simulation.
    inds_sub : 1-D array
        Indices to select a subsample of the N-body halos.
    seed : int
        Seed to generate random numbers.
    filters : tables.Filters
        Compression filters given by the PyTables package.
    comm : MPI communicator
        Use mpi4py to access a MPI communicator. If ``None``, the code is not
        parallelized.
    master_rank : int
        Index of the master rank.
    """
    redshifts = cntr.redshifts
    ages = cntr.ages
    if isinstance(cntr, DarkSeed):
        params_ag = {
            'm_min_cut': 0., 'm_max_cut': np.inf, 'eps_cut': 0.,
            'n_trail_cut': 0, 'eps_match':0, 'n_trail_match': 0,
            'n_max_trail': 1
        }
    elif params_ag is None:
        raise ValueError("When augmenting trees, params_ag should not be None.")
    #
    m_cut = params_ag['m_min_cut']
    #
    create_output(cntr, fname, filters, comm, master_rank)
    # Initialise new halos
    next_host_data = init_root_halos(cntr, mass_key, snap_max, root_halos)
    #
    pool_data, pool_list_chunk = None, [[] for i in range(snap_max + 1)]
    #
    for snap in range(snap_max, snap_min, -1):
        if comm is None or comm.Get_rank() == master_rank:
            logger.info(f"Processing snapshot {snap} ({memory_usage()})")

        host_data_raw, prog_data_raw = read_data(
            cntr, mass_key, snap, snap_max, inds_sub, comm, master_rank
        )
        host_chunk, prog_chunk = scatter_data(
            host_data_raw, prog_data_raw, next_host_data, pool_data,
            mass_key, m_cut, {}, comm, master_rank
        )
        z_root, t_root = redshifts[snap], ages[snap]
        z_target, t_target = redshifts[snap - 1], ages[snap - 1]
        new_prog_chunk, next_host_chunk, stats_chunk = augment_chunk(
            planter, host_chunk, prog_chunk, pool_list_chunk,
            mass_key, snap, z_root, t_root, z_target, t_target,
            redshifts, ages, m_cut, params_ag, comm
        )
        host_data, prog_data, next_host_data, pool_data, stats_data = gather_data(
            host_chunk, new_prog_chunk, next_host_chunk,
            pool_list_chunk[snap - 1], stats_chunk, comm, master_rank
        )
        save_data(
            cntr, fname, host_data, prog_data, stats_data,
            snap, snap_min,filters, comm, master_rank
        )

        if comm is None or comm.Get_rank() == master_rank:
            logger.info(dividing_line())


def create_partitions(cntr, mass_key, snap, n_par, by_forest_id=True):
    if by_forest_id:
        data = cntr.read_cols(snap, 'Host', cols=[mass_key, 'ForestID'])
        m_halo = data[mass_key]
        forest_id = data['ForestID']

        m_forest = {f_id: 0. for f_id in forest_id}
        for m_h, f_id in zip(m_halo, forest_id):
            m_forest[f_id] += m_h
        keys = np.asarray(list(m_forest.keys()))
        m_forest = np.asarray(list(m_forest.values()))
        keys = keys[m_forest.argsort()[::-1]]
        inds = np.zeros(len(keys), dtype='i4')
        for i_par in range(n_par):
            inds[i_par::n_par] = i_par
        inds_hash = {k: idx for k, idx in zip(keys, inds)}
        parts = [[] for i_par in range(n_par)]
        for idx, f_id in enumerate(forest_id):
            parts[inds_hash[f_id]].append(idx)
        parts = [np.asarray(p) for p in parts]
        return parts
    else:
        m_halo = cntr.read_cols(snap, 'Host', cols=[mass_key])[mass_key]

        inds_sorted = m_halo.argsort()[::-1]
        parts = [None]*n_par
        for i_par in range(n_par):
            parts[i_par] = inds_sorted[i_par::n_par]
        return parts


class DarkSeed:
    def __init__(self, emf, redshifts, box_size, z_grid=None):
        self.emf = emf
        self.z_grid = z_grid
        self.redshifts = redshifts
        self.ages = emf.cosmo.age(redshifts).to(units.megayear).value
        self.box_size = box_size


    def halo_kit(self):
        return HaloKit(self.emf, z_grid=self.z_grid)
