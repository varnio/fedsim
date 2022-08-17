#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="fedsim",
    version="0.6.1",
    license="Apache-2.0",
    description="Generic Federated Learning Simulator with PyTorch",
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="Farshid Varno",
    author_email="fr.varno@gmail.com",
    url="https://github.com/varnio/fedsim",
    packages=find_packages(),
    package_dir={"": "."},
    py_modules=[splitext(basename(path))[0] for path in glob("fedsim/*.py")]
    + [splitext(basename(path))[0] for path in glob("fedsim_cli/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://fedsim.varnio.com/",
        "Changelog": "https://fedsim.varnio.com/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/varnio/fedsim/issues",
    },
    install_requires=[
        "click",
        "numpy",
        "sklearn",
        "scikit-optimize",
        "tqdm",
        "torch",
        "torchvision",
        "tensorboard",
        "pyyaml",
        "logall",
    ],
    keywords="pytorch, neural networks, template, federated, federated \
        learning, deep learning, distributed learning",
    python_requires=">=3.6",
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "codecov",
        ]
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        "console_scripts": [
            "fedsim-cli = fedsim_cli.fedsim_cli:cli",
        ],
    },
)
