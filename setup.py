#!/usr/bin/env python

from setuptools import setup
from setuptools import find_namespace_packages

# Load the README file.
with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name='fedsim',
    author='Farshid Varno',
    author_email='f.varno@dal.ca',
    # Read this as
    #   - MAJOR VERSION 0
    #   - MINOR VERSION 1
    #   - MAINTENANCE VERSION 0
    version='0.1.0',
    description='Federated Learning Simulation Library!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/fvarno/fedsim',

    # These are the dependencies the library needs in order to run.
    install_requires=[
        'click',
        'numpy',
        'sklearn',
        'tqdm',
        'torch',
        'torchvision',
    ],
    keywords='pytorch, neural networks, template, federated, federated \
        learning, deep learning, distributed learning',
    packages=find_namespace_packages(),
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers', 'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology'
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
        'Programming Language :: Python :: 3'
    ],
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'fedsim = scripts.simulate:main',
        ],
    },
)
