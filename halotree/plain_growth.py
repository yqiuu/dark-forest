from .halotree import *

import numpy as np
from mpi4py import MPI


__all__ = ['PlainGrowth']


def _ScatterMass(mass, seed, comm, master_rank):
    if comm is None:
        return mass, seed
    else:
        i_rank = comm.Get_rank()
        n_rank = comm.Get_size()
        if i_rank == master_rank:
            n_mass = len(mass)
            count = [n_mass//n_rank]*n_rank
            count[master_rank] += n_mass%n_rank
            mass = np.asarray(mass, dtype = np.float64)
            seed = np.asarray(seed, dtype = np.uint64)
        else:
            count = None
        count = comm.bcast(count, root = master_rank)
        displ = np.append(0, np.cumsum(count)[:-1])
        mass_recv = np.zeros(count[i_rank], dtype = np.float64)
        seed_recv = np.zeros(count[i_rank], dtype = np.uint64)
        comm.Scatterv([mass, count, displ, MPI.DOUBLE], mass_recv, root = master_rank)
        comm.Scatterv([seed, count, displ, MPI.UNSIGNED_LONG], seed_recv, root = master_rank) 
        return mass_recv, seed_recv


def _GatherMass(mass, comm, master_rank):
    if comm is None:
        return mass
    else:
        i_rank = comm.Get_rank()
        count = comm.allgather(len(mass))
        displ = np.append(0, np.cumsum(count)[:-1])
        if i_rank == master_rank:
            mass_recv = np.zeros(np.sum(count), dtype = np.float64)
        else:
            mass_recv = np.array([])
        comm.Gatherv(
            [mass, count[i_rank]], [mass_recv, count, displ, MPI.DOUBLE], root = master_rank
        )
        return mass_recv


def PlainGrowth(
    halo_kit, mass, seed, z_root, z_target, m_res, comm = None, master_rank = 0, **params
):
    mass_rank, seed_rank = _ScatterMass(mass, seed, comm, master_rank)
    n_mass = len(mass_rank)
    mass_prog = np.empty(n_mass, dtype = object)
    for i_mass in range(n_mass):
        mass_prog[i_mass] = np.array([], dtype = 'f8')
    for i_mass in range(n_mass):
        tree = HaloMergeTree(
            halo_kit = halo_kit, m_root = mass_rank[i_mass], seed = seed_rank[i_mass],
            z_root = z_root, z_target = z_target, m_res = m_res, **params
        )
        mass_prog[i_mass] = tree.Halos(1)['mass']
    mass_next_rank = np.concatenate(mass_prog) 
    mass_next = _GatherMass(mass_next_rank, comm, master_rank)
    mass_next[::-1].sort()
    return mass_next

