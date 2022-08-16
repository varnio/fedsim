r"""
cli Top
-------
"""

import os

import click

from . import __version__
from .fed_learn import fed_learn
from .fed_tune import fed_tune

# Enable click
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"


@click.group()
@click.version_option(__version__)
def cli():
    pass


cli.add_command(fed_learn)
cli.add_command(fed_tune)


def main():
    cli()


if __name__ == "__main__":
    main()
