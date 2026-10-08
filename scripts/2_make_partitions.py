"""Split the catalogue into forest partitions using a YAML configuration argument."""

import sys
sys.path.append('/home/bsreedha/models/dark-forest/')

import numpy as np
import dforest

import yaml

inputs = yaml.safe_load(open(sys.argv[1]))
file_in = inputs['file_in']
#path_out = inputs['path_out']
mass_key = inputs['mass_key']

snap = inputs['snap_max']
n_par = 64

with dforest.DarkForestConnector(file_in) as cntr:
	parts = dforest.create_partitions(cntr, mass_key, snap, n_par)

for i, p in enumerate(parts):
	np.save('outputs/parts/L210_part%d.npy'%i, p)
