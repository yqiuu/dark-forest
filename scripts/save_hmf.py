"""Estimate and save a binned halo mass function from the configured simulation."""

import numpy as np
import sys

sys.path.append('/fred/oz013/balu/models/dark-forest/') 
import dforest


###L210N1536 box; SWIFT code
#fname = "/fred/oz009/GENESIS/L210_N1536_Grids_Halos_Trees/trees/VELOCIraptor.walkabletree.forestID.hdf5.0"
#snapshot = 119 # z = 5.007

###L105N2048 box; GADGET code
fname = "/fred/oz009/yqiu/L105_N2048/trees/VELOCIraptor.tree.walkabletree.forestID.hdf5"
snapshot = 77 # z = 5.0423

with dforest.VELOCIraptorConnector(fname) as cntr:
    mhaloes = cntr.__getitem__([snapshot, "Mass_FOF"])
    mhaloes = mhaloes*cntr.mass_conversion
    volume=cntr.volume * (cntr.little_h**3)

bins = dforest.smart_bins(mhaloes, 10000, 0.1, 'log')

counts = np.histogram(mhaloes, bins=bins)[0]
y = counts/np.diff(bins)/volume
x = np.sqrt(bins[:-1]*bins[1:])
y = y[:-1]
x = x[:-1]

#np.save("hmf_L210N1536_z5.npy", np.array([x, y]))

np.save("hmf_L105N2048_z5.npy", np.array([x, y]))
