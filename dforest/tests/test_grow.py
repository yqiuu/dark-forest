import os

import pytest
import numpy as np
import dforest
from numpy.testing import assert_allclose, assert_array_equal


def compare_dark_forest(dfor1, dfor2):
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
        # Compare host data
        host1 = dfor1.read_cols(i_snap, 'Host')
        host2 = dfor2.read_cols(i_snap, 'Host')
        assert_array_equal(host1.keys(), host2.keys())
        for k in host1.keys():
            if np.array_equal(np.round(host1[k]), host1[k]):
                assert_array_equal(host1[k], host2[k])
            else:
                assert_allclose(host1[k], host2[k])
        # Compare progenitor data
        if i_snap > 0:
            prog1 = dfor1.read_cols(i_snap, 'Prog')
            prog2 = dfor2.read_cols(i_snap, 'Prog')
            assert_array_equal(prog1.keys(), prog2.keys())
            for k in prog1.keys():
                if np.array_equal(np.round(prog1[k]), prog1[k]):
                    assert_array_equal(prog1[k], prog2[k])
                else:
                    assert_allclose(prog1[k], prog2[k])


def quick_plain_grow(fname, m_res, box_size, hmf_model='PS'):
    snap_min=0
    snap_max=20
    redshifts = np.linspace(10., 0., snap_max - snap_min + 1)

    seed = 20070831
    rstate = np.random.RandomState(seed)
    emf = dforest.EnhancedMassFunction(
        Mmin=np.log10(m_res) - .5, Mmax=15.5, hmf_model=hmf_model
    )
    volume = box_size**3
    root_halos = emf.sample_mass_func(m_res, 10**15, volume, rstate)
    dark_seed = dforest.DarkSeed(
        emf, redshifts, box_size, z_grid=np.linspace(0, 50, 1000)
    )
    planter = dforest.TreeBinarySplit(
        dark_seed, G0=0.57, gamma1=0.38, gamma2=-0.01,
        m_res=m_res, eps1=0.1, eps2=0.1, seed=seed
    )

    dforest.grow_halos(
        cntr=dark_seed,
        planter=planter,
        fname=fname,
        mass_key="Mvir",
        snap_max=snap_max,
        snap_min=snap_min,
        root_halos=root_halos,
    )


@pytest.fixture(scope="session")
def plain_growth_low_file(tmpdir_factory):
    box_size = 17
    m_res = 7.2e10
    fname = tmpdir_factory.mktemp("data").join("__plain_growth_low.hdf5")
    quick_plain_grow(str(fname), m_res, box_size, 'PS')
    return fname


def test_plain_growth_low(plain_growth_low_file):
    fname1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/test_plain_growth_low.hdf5"
    )
    fname2 = str(plain_growth_low_file)
    with dforest.DarkForestConnector(fname1) as dfor1, \
        dforest.DarkForestConnector(fname2) as dfor2:
        compare_dark_forest(dfor1, dfor2)


@pytest.fixture(scope="session")
def plain_growth_high_file(tmpdir_factory):
    box_size = 17
    m_res = 7.2e9
    fname = tmpdir_factory.mktemp("data").join("__plain_growth_high.hdf5")
    quick_plain_grow(str(fname), m_res, box_size, 'PS')
    return fname


def test_plain_growth_high(plain_growth_high_file):
    fname1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/test_plain_growth_high.hdf5"
    )
    fname2 = str(plain_growth_high_file)
    with dforest.DarkForestConnector(fname1) as dfor1, \
        dforest.DarkForestConnector(fname2) as dfor2:
        compare_dark_forest(dfor1, dfor2)
