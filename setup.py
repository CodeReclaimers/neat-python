# Installation script
from distutils.core import setup, Extension

setup(
    name='neat-python',
    version='0.2',
    author='cesar.gomes, mirrorballu2',
    author_email='nobody@nowhere.com',
    maintainer='CodeReclaimers, LLC',
    maintainer_email='alan@codereclaimers.com',
    url='https://github.com/CodeReclaimers/neat-python',
    license="BSD",
    description='A NEAT (NeuroEvolution of Augmenting Topologies) implementation',
    long_description='Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a method ' +
                     'developed by Kenneth O. Stanley for evolving arbitrary neural networks.',
    packages=['neat', 'neat/iznn', 'neat/nn', 'neat/ctrnn', 'neat/ifnn'],
    install_requires=['graphviz', 'python2-biggles', 'pp'],
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
