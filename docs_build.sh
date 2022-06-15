#!/bin/sh
cd docs
sphinx-apidoc -o ./source ..
make html

