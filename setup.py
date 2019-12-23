from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


exts = [Extension("dforest.halotree",
                  sources = ["dforest/clib/node_vector.c",
                             "dforest/clib/halo_relation.c",
                             "dforest/clib/tool_kit.c",
                             "dforest/clib/merge_tree.c",
                             "dforest/halotree.pyx"],
                  include_dirs = ["dforest/"],
                  extra_compile_args = ['-lgsl', '-lgslcblas'],
                  extra_link_args = ['-lgsl', '-lgslcblas'])]


for e in exts:
    e.cython_directives = {"embedsignature":True}

#exec(open('sector/version.py', 'r').read())

setup(name = 'halotree',
      ext_modules = exts,
      cmdclass = {'build_ext': build_ext},
)
