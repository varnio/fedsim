from click.testing import CliRunner

from fedsim_cli.fed_learn import fed_learn
from fedsim_cli.fed_tune import fed_tune
from fedsim_cli.fedsim_cli import cli


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, "--version")
    assert result.exit_code == 0


def test_fedlearn():
    runner = CliRunner()
    result = runner.invoke(fed_learn, "-r 0 --epochs 0 --device cpu")
    assert result.exit_code == 0


def test_fedtune():
    runner = CliRunner()
    result = runner.invoke(
        fed_tune,
        [
            "-r",
            "0",
            "--epochs",
            "0",
            "--local-optimizer",
            "SGD",
            "lr:Real:0-1",
            "--device",
            "cpu",
            "--n-iters",
            "2",
            "--skopt-n-initial-points",
            "2",
        ],
    )
    assert result.exit_code == 0
