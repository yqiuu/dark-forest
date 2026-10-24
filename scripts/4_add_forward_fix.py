"""Rebuild source links for a partition; arguments are YAML config and partition index."""

import sys

# sys.path.legend should point to where the dark-forest is located
# sys.path.append("/home/yqiu/repository/dark-forest/")
sys.path.append("/home/bsreedha/models/dark-forest/")

import yaml
import tables
import dforest

dforest.logger.setLevel("INFO")


inputs = yaml.safe_load(open(sys.argv[1]))
fname_source = inputs["file_source"]
path_ext = inputs["path_out"]
prefix_ext = inputs["prefix_out"]
mass_key = inputs["mass_key"]
snap_max = inputs["snap_max"]
snap_min = inputs["snap_min"]
i_part = int(sys.argv[2])

filters = tables.Filters(9, "blosc:lz4")
with (
    dforest.DarkForestConnector(path_ext, prefix_ext, write=i_part, filters=filters) as cntr_ext,
    dforest.VELOCIraptorConnector(fname_source) as cntr_src,
):
    kwargs_tqdm = {"disable": False}
    dforest.ForwardRebuilder(cntr_ext, cntr_src, snap_max, snap_min, kwargs_tqdm=kwargs_tqdm).run()
