import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from halotree import RandomStateGSL, Tree_Parkinson08, HaloKit, EnhancedMassFunction


def test_binary_spkit():
    # Create test tree
    hmf = EnhancedMassFunction(Mmin=6)
    halo_kit = HaloKit(hmf)
    rstate = RandomStateGSL(831)
    tree_test = Tree_Parkinson08(
        halo_kit=halo_kit, m_root=1e10, z_root=.5, z_target=[1., 2., 4.], rstate=rstate
    ).Halos(-1)
    # Load fiducial tree
    tree_fid = np.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_binary_split.npy')
    )
    # Compare
    assert_allclose(tree_test['z'], tree_fid['z'])
    assert_allclose(tree_test['mass'], tree_fid['mass'])
    assert_array_equal(tree_test['idx_tail'], tree_fid['idx_tail'])
    assert_array_equal(tree_test['idx_next'], tree_fid['idx_next'])
