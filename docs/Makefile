# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	@echo "[Makefile] cleaning..."
	@rm -rf "./$(BUILDDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html:
	# Note: this target has to be called "html" because its name is given as an argument to Sphinx
	@echo "[Makefile] generating HTML pages using sphinx-build..."
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

generate-api:
	@echo "[Makefile] generating API using sphinx-apidoc..."
	@sphinx-apidoc -fe --templatedir "$(SOURCEDIR)/_templates/apidoc" -o "$(SOURCEDIR)/reference" ../fedsim

all: clean generate-api html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
