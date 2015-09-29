# Installation script
from distutils.core import setup, Extension

setup(
    name='neat-python',
    version='0.2',
    author='cesar.gomes, mirrorballu2',
    maintainer='CodeReclaimers, LLC',
    maintainer_email='alan@codereclaimers.com',
    url='https://github.com/CodeReclaimers/neat-python',
    description='A NEAT (NeuroEvolution of Augmenting Topologies) implementation',
    packages=['neat', 'neat/iznn', 'neat/nn', 'neat/ctrnn', 'neat/ifnn'],
    requires=['graphviz', 'python2-biggles'],
    ext_modules=[
        Extension('neat/iznn/iznn_cpp', ['neat/iznn/iznn.cpp']),
        Extension('neat/nn/ann', ['neat/nn/nn_cpp/ANN.cpp', 'neat/nn/nn_cpp/PyANN.cpp']),
        Extension('neat/ifnn/ifnn_cpp', ['neat/ifnn/ifnn.cpp']), ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering'
    ]
)
