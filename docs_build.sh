#!/bin/sh
cd docs
sphinx-apidoc -o ./source ..
rm ./source/setup.rst
make html

