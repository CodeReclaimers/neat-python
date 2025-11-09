from setuptools import setup

setup(
    name='neat-python',
    version='1.0.0',
    author='Alan Mcintyre, Cesar Gomes Miguel, Carolina Feher da Silva, Marcio Lobo Netto',
    author_email='alan@codereclaimers.com',
    maintainer='Alan McIntyre',
    maintainer_email='alan@codereclaimers.com',
    url='https://github.com/CodeReclaimers/neat-python',
    license="BSD",
    description='A NEAT (NeuroEvolution of Augmenting Topologies) implementation',
    long_description='Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a method ' +
                     'developed by Kenneth O. Stanley for evolving arbitrary neural networks. ' +
                     'Version 1.0.0 includes full innovation number tracking as described in the original NEAT paper.',
    long_description_content_type= 'text/x-rst',
    packages=['neat', 'neat/iznn', 'neat/nn', 'neat/ctrnn'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ]
)
