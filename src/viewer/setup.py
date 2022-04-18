
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize 
import numpy as np




module1 = Extension("libintersect", sources=["libintersect.pyx", "intersect_mesh_ray.cpp"], include_dirs=['.', np.get_include()], extra_link_args=['-fopenmp'], extra_compile_args=['--std=c++11','-fopenmp']) 

setup(
	name = "libintersect",
	ext_modules = cythonize([ module1])
)
