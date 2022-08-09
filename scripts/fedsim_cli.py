r"""
cli Top
-------
"""

import os

import click

from .fed_learn import fed_learn
from .fed_tune import fed_tune

# Enable click
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"


@click.group()
def cli():
    pass


cli.add_command(fed_learn)
cli.add_command(fed_tune)


if __name__ == "__main__":
    cli()
