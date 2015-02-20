from distutils.core import setup
from Cython.Build import cythonize

setup(name='Dynmodel',
      version='0.2',
      description='Helper for creating dynamic state variable models',
      author='Greg Baker',
      author_email='ggbaker@sfu.ca',
      packages=['dynmodel'],
      ext_modules = cythonize('dynmodel/interpolatex.pyx'),
     )