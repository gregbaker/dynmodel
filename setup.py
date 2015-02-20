from setuptools import setup
try:
    from Cython.Build import cythonize
    have_cython = True
except ImportError:
    have_cython = False


if have_cython:
    ext_modules = cythonize('dynmodel/interpolatex.pyx')
else:
    ext_modules = []

setup(name='Dynmodel',
      version='0.2',
      description='Helper for creating dynamic state variable models',
      author='Greg Baker',
      author_email='ggbaker@sfu.ca',
      license='GPL2',
      packages=['dynmodel'],
      install_requires=['pyconfig'],
      ext_modules = ext_modules,
     )