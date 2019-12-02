import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from halotree import RandomStateGSL, Tree_Parkinson08, HaloKit, GrowKit, EnhancedMassFunction


def test_binary_spkit():
    # Create test tree
    hmf = EnhancedMassFunction(Mmin=6)
    halo_kit = HaloKit(hmf)
    rstate = RandomStateGSL(831)
    grow_kit = GrowKit(G0=0.57, gamma1=0.38, gamma2=-0.01, m_res=1e8, eps1=0.1, eps2=0.1)
    tree_test = Tree_Parkinson08(
        halo_kit=halo_kit, grow_kit=grow_kit, m_root=1e10,
        z_root=.5, z_target=[1., 2., 4.], rstate=rstate
    ).Halos(-1)
    # Load fiducial tree
    tree_fid = np.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/test_binary_split.npy')
    )
    # Compare
    assert_allclose(tree_test['z'], tree_fid['z'])
    assert_allclose(tree_test['mass'], tree_fid['mass'])
    assert_array_equal(tree_test['idx_tail'], tree_fid['idx_tail'])
    assert_array_equal(tree_test['idx_next'], tree_fid['idx_next'])
