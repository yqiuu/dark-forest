from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


exts = [Extension("planter.planter",
                  sources = ["planter/clib/planter.c",
                             "planter/planter.pyx"],
                  include_dirs = ["planter/"],
                  extra_compile_args = ['-lgsl', '-lgslcblas'],
                  extra_link_args = ['-lgsl', '-lgslcblas'])]


for e in exts:
    e.cython_directives = {"embedsignature":True}

#exec(open('sector/version.py', 'r').read())

setup(name = 'planter',
      ext_modules = exts,
      cmdclass = {'build_ext': build_ext},
)
