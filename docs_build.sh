#!/bin/sh
cd docs
sphinx-apidoc -fMeT ../fedsim -o ./source/reference -d 1 -t ./source/_templates/apidoc
make clean
make html
