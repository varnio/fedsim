# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import unicode_literals

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../sphinxext"))

# -- Project information -----------------------------------------------------
master_doc = "index"
project = "fedsim"
copyright = "2022, Farshid Varno"
year = "2022"
author = "Farshid Varno"
copyright = "{0}, {1}".format(year, author)
version = release = "0.1.4"
# The full version, including alpha/beta/rc tags

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "sphinx_panels",
]

add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# This is the default encoding, but it doesn't hurt to be explicit
source_encoding = "utf-8"

# The toplevel toctree document (renamed to root_doc in Sphinx 4.0)
root_doc = master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "**.ipynb_checkpoints", "_*"]


# GitHub extension

github_project_url = "https://github.com/varnio/fedsim/"

html_css_files = [
    "fsm.css",
]

html_theme = "pydata_sphinx_theme"
# is_release_build = True
html_logo = "_static/logo.png"
# include_analytics = is_release_build
# if include_analytics:
#     html_theme_options["google_analytics_id"] = "XXX"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_use_smartypants = True

# this makes this the canonical link for all the pages on the site...
html_baseurl = "https://fedsim.varnio.com/en/latest/"

html_last_updated_fmt = "%b %d, %Y"

# Content template for the index page.
html_index = "index.html"

html_split_index = False

html_context = {"default_mode": "light"}

# Prevent sphinx-panels from loading bootstrap css, the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"


html_sidebars = {
    "index": ["search-field"],
    "contribute": ["notes/contributing"],
    # "demo/no-sidebar": [],  # Test what page looks like with no sidebar items
}


html_theme_options = {
    "external_links": [
        {
            "url": "https://github.com/varnio/fedsim/releases",
            "name": "Changelog",
        },
        {
            "url": "https://varnio.com",
            "name": "Varnio",
        },
        {
            "url": "https://www.buymeacoffee.com/fvarno",
            "name": "Donate to FedSim",
        },
    ],
    "github_url": "https://github.com/varnio/fedsim",
    "twitter_url": "https://twitter.com/VarnioTech",
    # "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/fedsim",
            "icon": "fas fa-box",
        },
        {
            "name": "Varnio",
            "url": "https://varnio.com",
            "icon": "_static/varnio-logo.png",
            "type": "local",
        },
    ],
    "logo": {
        "text": "FedSim",
        "image_dark": "logo-light.png",
    },
    "show_toc_level": 1,
    "navbar_start": ["navbar-logo"],
}


# Prevent sphinx-panels from loading bootstrap css, the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Copies only relevant code, not the '>>>' prompt
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# If true, add an index to the HTML documents.
html_use_index = False

# If true, generate domain-specific indices in addition to the general index.
# For e.g. the Python domain, this is the global module index.
html_domain_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
# html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.
html_use_opensearch = "False"

# Output file base name for HTML help builder.
htmlhelp_basename = "FedSimdoc"

# Use typographic quote characters.
smartquotes = False

# Path to favicon
html_favicon = "_static/logo.png"


html_short_title = "%s-%s" % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
