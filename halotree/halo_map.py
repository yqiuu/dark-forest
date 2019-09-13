import numpy as np


def Move(workspace, inds0, inds1, it, centre, grid_info, sub_info):
    #
    pos0 = workspace['pos0']
    pos1 = workspace['pos1']
    moved = workspace['moved']
    idx_dim = workspace['order'][it]
    eps = workspace['eps']
    #
    b_mid =  centre[idx_dim]
    x0 = pos0[inds0, idx_dim]
    cond0 = x0 < b_mid
    n_left0 = np.count_nonzero(cond0)
    x1 = pos1[inds1, idx_dim]
    cond1 = x1 < b_mid
    n_left1 = np.count_nonzero(cond1)
    n_move = n_left1 - n_left0
    if n_move > 0:
        # Move from right to left
        inds_sorted = np.argsort(x0 - b_mid)
        inds_move = inds_sorted[n_left0 : n_move+n_left0]
        cond0[inds_move] = True
        inds_move = inds0[inds_move]
        pos0[inds_move, idx_dim] = (1 - eps)*b_mid
        moved[inds_move] = True
    else:
        # Move from left to right
        n_right0 = len(x0) - n_left0
        inds_sorted = np.argsort(b_mid - x0)
        inds_move = inds_sorted[n_right0 : n_right0-n_move]
        cond0[inds_move] = False
        inds_move = inds0[inds_move]
        pos0[inds_move, idx_dim] = (1 + eps)*b_mid
        moved[inds_move] = True
  
    it += 1
    if it != 3:
        grid_info_left = grid_info.copy()
        grid_info_left[idx_dim] = 0
        Move(workspace, inds0[cond0], inds1[cond1], it, centre, grid_info_left, sub_info)
        grid_info_right = grid_info.copy()
        grid_info_right[idx_dim] = 1
        Move(workspace, inds0[~cond0], inds1[~cond1], it, centre, grid_info_right, sub_info)
    else:
        size = .5*workspace['size']
        grid_info_left = np.array(grid_info)
        grid_info_left[idx_dim] = 0
        centre_left = centre + size*(grid_info_left - 0.5)
        inds_left0 = inds0[cond0]
        inds_left1 = inds1[cond1]
        if len(inds_left0) > 0:
            sub_info.append((inds_left0, inds_left1, centre_left))
        grid_info_right = np.array(grid_info)
        grid_info_right[idx_dim] = 1
        centre_right = centre + size*(grid_info_right - 0.5)
        inds_right0 = inds0[~cond0]
        inds_right1 = inds1[~cond1]
        if len(inds_right0) > 0:
            sub_info.append((inds_right0, inds_right1, centre_right))


def FindMatch(pos0, pos1, n_grid, box_size, rstate, eps=1e-6):
    n_pts = len(pos0)
    sub_info = [(np.arange(n_pts), np.arange(n_pts), [.5*box_size]*3)]
    workspace = {
        'pos0': pos0,
        'pos1': pos1,
        'moved': np.full(n_pts, False),
        'order': [0, 1, 2],
        'size': box_size,
        'eps': eps
    }

    i_grid = 1
    while i_grid < n_grid:
        sub_info_next = []
        for inds0, inds1, centre in sub_info:
            rstate.shuffle(workspace['order'])
            Move(workspace, inds0, inds1, 0, centre, [-1]*3, sub_info_next)
        sub_info = sub_info_next
        workspace['size'] *= .5
        i_grid *= 2
    return workspace['moved']
