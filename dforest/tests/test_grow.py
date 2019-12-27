import os

import pytest
import numpy as np
import dforest
from numpy.testing import assert_allclose, assert_array_equal


def CompareDarkForest(dfor1, dfor2):
    # Compare attributes
    assert_allclose(dfor1.little_h, dfor2.little_h)
    assert_allclose(dfor1.omega_m, dfor2.omega_m)
    assert_allclose(dfor1.sigma_8, dfor2.sigma_8)
    assert_allclose(dfor1.n_spectral, dfor2.n_spectral)
    assert_allclose(dfor1.mass_unit, dfor2.mass_unit)
    assert_allclose(dfor1.radius_unit, dfor2.radius_unit)
    assert_allclose(dfor1.velocity_unit, dfor2.velocity_unit)
    assert_allclose(dfor1.m_particle, dfor2.m_particle)
    # Compare redshifts
    assert_allclose(dfor1.redshifts, dfor2.redshifts)
    #
    n_snap = len(dfor1.redshifts)
    for i_snap in range(n_snap):
        dfor1.SwitchSnapshot(i_snap)
        dfor2.SwitchSnapshot(i_snap)
        # Compare host data
        host1 = dfor1.Read('Host')
        host2 = dfor2.Read('Host')
        assert_array_equal(host1.keys(), host2.keys())
        for k in host1.keys():
            if np.array_equal(np.round(host1[k]), host1[k]):
                assert_array_equal(host1[k], host2[k])
            else:
                assert_allclose(host1[k], host2[k])
        # Compare progenitor data
        if i_snap > 0:
            prog1 = dfor1.Read('Prog')
            prog2 = dfor2.Read('Prog')
            assert_array_equal(prog1.keys(), prog2.keys())
            for k in prog1.keys():
                if np.array_equal(np.round(prog1[k]), prog1[k]):
                    assert_array_equal(prog1[k], prog2[k])
                else:
                    assert_allclose(prog1[k], prog2[k])


def QuickPlainGrowth(fname, m_res, box_size, hmf_model='PS'):
    snap_min=0
    snap_max=20
    redshifts = np.linspace(10., 0., snap_max - snap_min + 1)

    hmf = dforest.EnhancedMassFunction(Mmin=np.log10(m_res) - .5, Mmax=15.5, hmf_model=hmf_model)
    halo_kit = dforest.HaloKit(hmf, z_grid=np.linspace(0, 50, 1000))
    grow_kit = dforest.GrowKit(m_res=m_res, G0=0.57, gamma1=0.38, gamma2=-0.01, eps1=0.1, eps2=0.1)
    volume = box_size**3

    seed = 20070831
    rstate = np.random.RandomState(seed)
    root_halos = hmf.SampleMassFunction(m_res, 10**15, volume, rstate)
    dark_seed = dforest.DarkSeed(hmf, redshifts, box_size)

    dforest.GrowHalos(
        cntr=dark_seed,
        fname=fname,
        mass_key="Mvir",
        halo_kit=halo_kit,
        grow_kit=grow_kit,
        snap_max=snap_max,
        snap_min=snap_min,
        root_halos=root_halos,
        seed=seed,
    )


@pytest.fixture(scope="session")
def plain_growth_low_file(tmpdir_factory):
    box_size = 17
    m_res = 7.2e10
    fname = tmpdir_factory.mktemp("data").join("__plain_growth_low.hdf5")
    QuickPlainGrowth(str(fname), m_res, box_size, 'PS')
    return fname


def test_plain_growth_low(plain_growth_low_file):
    fname1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/test_plain_growth_low.hdf5"
    )
    fname2 = str(plain_growth_low_file)
    with dforest.DarkForestConnector(fname1) as dfor1, \
        dforest.DarkForestConnector(fname2) as dfor2:
        CompareDarkForest(dfor1, dfor2)


@pytest.fixture(scope="session")
def plain_growth_high_file(tmpdir_factory):
    box_size = 17
    m_res = 7.2e9
    fname = tmpdir_factory.mktemp("data").join("__plain_growth_high.hdf5")
    QuickPlainGrowth(str(fname), m_res, box_size, 'PS')
    return fname


def test_plain_growth_high(plain_growth_high_file):
    fname1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/test_plain_growth_high.hdf5"
    )
    fname2 = str(plain_growth_high_file)
    with dforest.DarkForestConnector(fname1) as dfor1, \
        dforest.DarkForestConnector(fname2) as dfor2:
        CompareDarkForest(dfor1, dfor2)

