import os

import numpy as np
import dforest
from numpy.testing import assert_allclose, assert_array_equal


def test_binary_spkit():
    # Create test tree
    emf = dforest.EnhancedMassFunction(Mmin=6)
    halo_kit = dforest.HaloKit(emf)
    grower = dforest.TreeBinarySplit(
        halo_kit, G0=0.57, gamma1=0.38, gamma2=-0.01,
        m_res=1e8, eps1=0.1, eps2=0.1, seed=831
    )
    m_root = 1e10
    z_root = 0.5
    z_target = np.array([1., 2., 4.])
    tree_test = grower.grow(
        m_root=m_root, z_root=z_root, z_target=z_target
    ).tree()
    # Load fiducial tree
    fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data/test_binary_split.npy'
    )
    tree_fid = np.load(fname)
    redshifts = np.append(z_root, z_target)
    # Compare
    assert_allclose(redshifts[tree_test['level']], tree_fid['z'])
    assert_allclose(tree_test['mass'], tree_fid['mass'])
    assert_array_equal(tree_test['idx_tail'], tree_fid['idx_tail'])
    assert_array_equal(tree_test['idx_next'], tree_fid['idx_next'])
