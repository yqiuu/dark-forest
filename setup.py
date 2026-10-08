from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


exts = [
    Extension(
        "dforest._dtree",
        sources=[
            "dforest/dtree/node_vector.c",
            "dforest/dtree/halo_relation.c",
            "dforest/dtree/tool_kit.c",
            "dforest/dtree/merge_tree.c",
            "dforest/_dtree.pyx",
        ],
        include_dirs=["dforest/"],
        extra_compile_args=["-lgsl", "-lgslcblas"],
        extra_link_args=["-lgsl", "-lgslcblas"],
    )
]


for e in exts:
    e.cython_directives = {"embedsignature": True}

# exec(open('sector/version.py', 'r').read())

setup(
    name="halotree",
    ext_modules=exts,
    cmdclass={"build_ext": build_ext},
)
