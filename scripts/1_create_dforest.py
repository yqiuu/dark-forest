"""Convert the configured VELOCIraptor host trees to Dark Forest HDF5."""

import sys
import logging

# sys.path.legend should point to where the dark-forest is located
sys.path.append("/home/bsreedha/models/dark-forest/")
logging.getLogger("dforest").setLevel(logging.INFO)

import dforest  # noqa: E402 -- Configure logging before loading the package.

## fname points to the merger tree file
##GADGET boxes:
# fname = "/fred/oz009/pelahi/GENESIS/L105_N2048/newtree/VELOCIraptor.tree.walkabletree.forestID.hdf5"
# fname = "/fred/oz009/pelahi/GENESIS/L105_N2048/tree.old/VELOCIraptor.tree.t4.0-189.walkabletree.forestID.hdf5"
# fname = "/fred/oz009/pelahi/GENESIS/L35_N2650/tree/VELOCIraptor.tree.t4.0-99.walkabletree.forestID.hdf5"
# fname = "/fred/oz009/yqiu/L105_N2048/trees/VELOCIraptor.tree.walkabletree.forestID.hdf5"

##SWIFT boxes:
# fname = "/fred/oz070/smutch/test_boxes/L10N128/meraxes/trees/VELOCIraptor.walkabletree.forestID.hdf5"
# fname = "/fred/oz009/GENESIS/L210_N1536_Grids_Halos_Trees/trees/VELOCIraptor.walkabletree.forestID.hdf5.0"
# fname = "/fred/oz009/GENESIS/L210_N3072_Grids_Halos_Trees-smutch/trees/VELOCIraptor.walkabletree.forestID.hdf5.0"
fname = "/fred/oz009/GENESIS/L210_N4320_Grids_Halos_Trees-balu/trees/VELOCIraptor.walkabletree.forestID.hdf5.0"

save_name = "./outputs/dforest_out.hdf5"

snap_min = 11  # first snapshot where the haloes start appearing in the Nbody merger tree. dfrorest breaks if the snap_min has no haloes.
snap_max = 119  # your smallest snapshot of interest; usually the last snapshot of the tree.

with dforest.VELOCIraptorConnector(fname) as cntr:
    dforest.DarkForestSaver(
        cntr,
        save_name,
        ["Mass_200mean", "Mass_200crit", "Mass_FOF", "Mass_tot"],
        ["ForestID"],
        snap_max,
        snap_min,
    ).run()
