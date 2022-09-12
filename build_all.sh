#!/bin/sh
# black --line-length 79 .
python setup.py bdist_wheel sdist

# remember:
# - copy README.md content to quick user guide in docs but change relative paths to
# replace ../_static/ instead of docs/source/_static/ wherever applies.
