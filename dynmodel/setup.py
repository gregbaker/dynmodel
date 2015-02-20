from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'dynmodel-cython-code',
  ext_modules = cythonize("interpolatex.pyx", language="c++"),
)
