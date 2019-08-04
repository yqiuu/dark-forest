import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def InheritSpin(pool, prog):
    m_max = -1.
    spin = np.nan
    for p_id, _, m in prog:
        if m > m_max:
            m_max = m
            spin = pool[p_id]
        del pool[p_id]
    return spin


def AddSpin(
    handler, hc_path, hc_prefix, snap_min, snap_max, spin_key, spin_sampler,
    progress_bar = True, kwargs_tqdm = {}
):
    redshifts = handler.redshifts
    pool = {}
    # Create progress bar
    snap_range = range(snap_min, snap_max + 1)
    if progress_bar:
        snap_range = tqdm(snap_range, desc = "Adding spin parameters", **kwargs_tqdm)
    for snap in snap_range:
        z = redshifts[snap]
        handler.CacheSnapshot(snap)
        with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap))) as store:
            df_host = pd.read_hdf(store, 'Host')
            halo_id = np.array(handler['ID'], dtype = df_host.index.dtype)
            host_id = halo_id[handler['HostHaloID'] == handler.HostID()]
            df_spin = pd.DataFrame(handler[spin_key], index = halo_id, columns = [spin_key])

            halo_id_sim = df_host.index[df_host.index >= 0]
            # Add spin parameter for halos in the simulation
            df_spin.drop(
                index = np.setdiff1d(host_id, halo_id_sim, assume_unique = True), inplace = True
            )
            # Add spin parameter for MC halos
            halo_id_mc = df_host.index[df_host.index < 0]
            spin = np.zeros(len(halo_id_mc))
            if snap > snap_min:
                prog_pointer = df_host.loc[halo_id_mc, ['ProgIndex', 'ProgNumber']].values
                list_prog = list(pd.read_hdf(store, 'Prog').to_records())

                spin = np.zeros(len(halo_id_mc))
                cond = prog_pointer[:, -1] > 0
                spin[cond] = [
                    InheritSpin(pool, list_prog[idx:idx + num]) for idx, num in prog_pointer[cond]
                ]
                cond = ~cond
            else:
                cond = np.full(len(spin), True)
            spin[cond] = spin_sampler.Draw(sum(cond), z)
            # Save output
            df_spin = df_spin.append(pd.DataFrame(spin, index = halo_id_mc, columns = [spin_key]))
            df_spin.to_hdf(store, 'Spin')
            #
            host_id = df_host.index
            pool.update({h_id:s for h_id, s in zip(host_id, df_spin.loc[host_id, spin_key])})


