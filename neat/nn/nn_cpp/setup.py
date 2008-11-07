# Installation script
from distutils.core import setup, Extension
setup(
      name='neat-python',      
      packages=['nn_cpp'],
      ext_modules=[               
               Extension('ann', ['ANN.cpp', 'PyANN.cpp']),],
)
