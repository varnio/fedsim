import os
import click
from fed_learn import fed_learn

# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


@click.group()
def run():
    pass


run.add_command(fed_learn)


def main():
    """ main fn

    """
    run()


if __name__ == '__main__':
    main()
