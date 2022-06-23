#!/bin/sh
cd docs
sphinx-apidoc -o ./source ..
rm ./source/setup.rst
rm ./source/scripts.rst
make clean
make html
