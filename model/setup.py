# Example setup.py for building a Cython extension
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("my_cython_module.pyx"),
)