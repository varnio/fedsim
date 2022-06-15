cd docs
sphinx-apidoc -o . ..
rm source/setup.rst
make html

