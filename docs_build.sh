#!/bin/sh
cd docs
sphinx-apidoc -fMeT ../fedsim -o ./source/reference
make clean
make html
