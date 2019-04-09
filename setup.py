from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


exts = [Extension("halotree.halotree",
                  sources = ["halotree/clib/node_vector.c",
                             "halotree/clib/halo_relation.c",
                             "halotree/clib/tool_kit.c",
                             "halotree/clib/merge_tree.c",
                             "halotree/halotree.pyx"],
                  include_dirs = ["planter/"],
                  extra_compile_args = ['-lgsl', '-lgslcblas'],
                  extra_link_args = ['-lgsl', '-lgslcblas'])]


for e in exts:
    e.cython_directives = {"embedsignature":True}

#exec(open('sector/version.py', 'r').read())

setup(name = 'halotree',
      ext_modules = exts,
      cmdclass = {'build_ext': build_ext},
)
