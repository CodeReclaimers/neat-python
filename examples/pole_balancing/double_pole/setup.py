# Compile the C++ Python extension for the
# cart-pole experiment:
# python setup.py build_ext -i
from distutils.core import setup, Extension
setup(
      name='Cart-pole experiment',
      ext_modules=[
               Extension('dpole', ['dpole.cpp'])]               
)
