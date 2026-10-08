import os
import sys

#sys.path.legend should point to where the dark-forest is located
sys.path.append('/home/bsreedha/models/dark-forest/') 

import yaml
import dforest
dforest.logger.setLevel("INFO")


inputs = yaml.safe_load(open(sys.argv[1]))
fname_src = inputs['file_source']
path_ext = inputs['path_out']
prefix_ext = inputs['prefix_out']
mass_key = inputs['mass_key']
snap_max = inputs['snap_max']
snap_min = inputs['snap_min']

save_name = os.path.join(path_ext, f'tree_{snap_min}_{snap_max}.hdf5')
with dforest.DarkForestConnector(path_ext, prefix=prefix_ext) as cntr_ext, \
    dforest.VELOCIraptorConnector(fname_src) as cntr_src:
    m_particle = cntr_src.m_particle
    dforest.VELOCIraptorSaver(
        cntr_ext=cntr_ext, 
        cntr_src=cntr_src,
        save_name=save_name, 
        items_mass=[
            ('Mass_tot', 'Mass_FOF'),
            ('Mass_200crit', 'Mass_FOF'),
        ],
        items_norm=[
            'ID',
            ('hostHaloID', 'HostHaloID'),
            ('Head', 'Descendant'),
            ('Tail', 'ID'),
            'ForestID',
            ('R_200crit', 'R_200crit_FOF'),
            ('lambda_B', 'Spin_200crit_FOF'),
            ('Vmax', 'V_200crit_FOF'),
            'Lx', 'Ly', 'Lz',
            'Xc', 'Yc', 'Zc',
            ('VXc', 'Xc', lambda x: 0.*x),
            ('VYc', 'Xc', lambda x: 0.*x),
            ('VZc', 'Xc', lambda x: 0.*x),
            ('npart', 'Mass_FOF', lambda x: x/m_particle)
        ],
        snap_max=snap_max,
        snap_min=snap_min,
    ).run()
