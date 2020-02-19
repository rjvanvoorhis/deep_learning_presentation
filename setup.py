from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="matrix_math", sources=["src/cython_utils.pyx"])
setup(ext_modules=cythonize(ext))
