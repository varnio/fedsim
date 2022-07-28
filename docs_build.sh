#!/bin/sh
cd docs
sphinx-apidoc -fMeT ../fedsim -o ./source/reference -d 1
python rst_process.py
make clean
make html
