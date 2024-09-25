from Cython.Build import cythonize
from distutils.core import setup

setup(ext_modules=cythonize("cythonfn.pyx", compiler_directives={"language_level": "3"}))
