from setuptools import setup

setup(
    name='ganariya-ne',
    version='0.0.1',
    author='cesar.gomes, mirrorballu2, ganariya',
    author_email='nobody@nowhere.com',
    maintainer='CodeReclaimers, LLC',
    maintainer_email='ganariya2525@gmail.com',
    url='https://github.com/ganariya/neat-python',
    license="BSD",
    description='A NEAT (NeuroEvolution of Augmenting Topologies) implementation',
    long_description='Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a method ' +
                     'developed by Kenneth O. Stanley for evolving arbitrary neural networks.',
    packages=['neat', 'neat/iznn', 'neat/nn', 'neat/ctrnn'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering'
    ]
)
