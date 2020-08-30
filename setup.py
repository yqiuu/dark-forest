from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


exts = [
    Extension(
        "dforest._dtree",
        sources=[
            "dforest/dtree-cpp/nut/nut.cpp",
            "dforest/dtree-cpp/halo_relation.cpp",
            "dforest/dtree-cpp/binary_split.cpp",
            "dforest/dtree-cpp/ensemble.cpp",
            "dforest/_dtree.pyx"
        ],
        include_dirs=["dforest"],
        extra_compile_args=['-lgsl', '-lgslcblas'],
        extra_link_args=['-lgsl', '-lgslcblas'],
        language="c++"
    )
]


for e in exts:
    e.cython_directives = {"embedsignature":True}

#exec(open('sector/version.py', 'r').read())

setup(
    name='dforest',
    ext_modules=exts,
    cmdclass={'build_ext': build_ext},
)
