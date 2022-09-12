FedSim
======

.. image:: https://github.com/varnio/fedsim/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/varnio/fedsim/actions

.. image:: https://img.shields.io/pypi/v/fedsim.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/fedsim

.. image:: https://readthedocs.org/projects/fedsim/badge/?version=stable
    :target: https://fedsim.readthedocs.io/en/latest/?badge=stable

.. image:: https://img.shields.io/pypi/wheel/fedsim.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/fedsim

.. image:: https://img.shields.io/pypi/pyversions/fedsim.svg
    :alt: Supported versions
    :target: https://pypi.org/project/fedsim

.. image:: https://img.shields.io/pypi/implementation/fedsim.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/fedsim

.. image:: https://codecov.io/gh/varnio/fedsim/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/varnio/fedsim

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://badges.gitter.im/varnio/community.svg
    :alt: Gitter
    :target: https://gitter.im/varnio/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


FedSim is a comprehensive and flexible Federated Learning Simulator! It aims to provide the researchers with an easy to develope/maintain simulator for Federated Learning.
See documentation at `here <https://fedsim.varnio.com/en/latest/>`_!


Installation
============

.. code-block:: bash

   pip install fedsim

That's it! You are all set!

-------------------


Design Architecture
===================

.. image:: https://raw.githubusercontent.com/varnio/fedsim/3387a994664853c599094a72b342b8f7f3dba0f2/docs/source/_static/arch.svg
    :width: 90%



CLI
=====

Minimal example
---------------

Fedsim provides powerful cli tools that allow you to focus on designing what is truly important.
Simply enter the following command to begin federatively training a model.

.. code-block:: bash

    fedsim-cli fed-learn

The "MNIST" dataset is partitioned on 500 clients by default, and the FedAvg algorithm is used to train a minimal model with two fully connected layers.
A text file is made that descibes the configuration for the experiment and a summary of results when it is finished. Additionally, a tensorboard log file is made to monitor the scores/metrics of the training.
The directory that these files are stored is (reconfigurable and is) displayed while the experiment is running.

.. image:: https://github.com/varnio/fedsim/blob/main/docs/source/_static/examples/one_line_train.gif?raw=true

Hooking scores to cli tools
---------------------------

In case you are interested in a certain metric you can make a query for it in your command.
For example, lets assume we would like to test and report:
* the accuracy score of the global model on global test dataset both every 21 rounds and every 43 rounds.
* the average accuracy score of the local models every 15 rounds.
Here's how we modify the above command:

.. code-block:: bash

    fedsim-cli fed-learn \
        --global-score Accuracy score_name:acc21 split:test log_freq:21 \
        --global-score Accuracy score_name:acc43 split:test log_freq:43 \
        --local-score Accuracy split:train log_freq:15

.. image:: https://github.com/varnio/fedsim/blob/main/docs/source/_static/examples/add_metrics.gif?raw=true

.. image:: https://github.com/varnio/fedsim/blob/main/docs/source/_static/examples/tb_ex.png?raw=true

Check `Fedsim Scores Page <https://fedsim.varnio.com/en/latest/reference/fedsim.scores.html>`_ for the list of all other scores like Accyracy or define your custom score.

Changing the Data
-----------------

Data partitioning and retrieval is controlled by a ``DataManager`` object. This object could be controlled through `-d` or `--data-manager` flag in most cli commands.
In the following we modify the arguments of the default ``DataManager`` such that ``CIFAR100`` is partitioned over 1000 clients.

.. code-block:: bash

    fedsim-cli fed-learn \
        --data-manger BasicDataManager dataset:cifar100 num_partitions:1000 \
        --num-clients 1000 \
        --model SimpleCNN2 num_classes:100 \
        --global-score Accuracy split:test log_freq:15

Notice that we also changed the model from default to ``SimpleCNN2`` which by default takes 3 input channels.
You can learn about existing data managers at `data manager documentation <https://fedsim.varnio.com/en/latest/reference/fedsim.distributed.data_management.html>`_ and Custom data managers at `this guide to make Custom data managers <https://fedsim.varnio.com/en/latest/user/data_manager.html>`_.

.. note::

    Arguments of the constructor of any component (rectangular boxes in the image of design architecture) could be given in `arg:value` format following its name (or `path` if a local file is provided).
    Among these component, the algorithm is special, in that the arguments are controlled internally. The only arguments of the algorithm object that could be directly controlled in your commands is the algorithm specific ones (mostly hyper-parameters).
    Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --algorithm AdaBest mu:0.01 beta:0.6 ...


Feed CLI with Customized Components
-----------------------------------

The cli tool can take a locally defined component by ingesting its path.
For example, to automatically include your custom algorithm by the a command of the cli tool, you can place your class in a python file and pass the path of the file to `-a` or `--algorithm` option (without .py) followed by colon and name of the algorithm definition (class or method).
For instance, if you have algorithm `CustomFLAlgorithm` stored in a `foo/bar/my_custom_alg.py`, you can pass `--algorithm foo/bar/my_custom_alg:CustomFLAlgorithm`.


.. code-block:: bash

        fedsim-cli fed-learn --algorithm foo/bar/my_custom_alg_file:CustomFLAlgorithm mu:0.01 ...

The same is possible for any other component, for instance for a Custom model:

.. code-block:: bash

        fedsim-cli fed-learn --model foo/bar/my_model_file:CustomModel num_classes:1000 ...


More about cli commands
-----------------------

For help with cli check `fedsim-cli documentation <https://fedsim.varnio.com/en/latest/clidoc/index.html>`_ or read the output of the following commands:

.. code-block:: bash

   fedsim-cli --help
   fedsim-cli fed-learn --help
   fedsim-cli fed-tune --help

Python API
==========

Fedsim is shipped with some of the most well-known Federated Learning algorithms included. However, you will most likely need to quickly develop and test your custom algorithm, model, data manager, or score class.
Fedsim has been designed in such a way that doing all of these things takes almost no time and effort. Let's start by learning how to import and use Fedsim, and then we'll go over how to easily modify existing modules and classes to your liking.
Check the following basic example:

.. code-block:: python

    from logall import TensorboardLogger
    from fedsim.distributed.centralized.training import FedAvg
    from fedsim.distributed.data_management import BasicDataManager
    from fedsim.models import SimpleCNN2
    from fedsim.losses import CrossEntropyLoss
    from fedsim.scores import Accuracy

    n_clients = 1000

    dm = BasicDataManager("./data", "cifar100", n_clients)
    sw = TensorboardLogger(path=None)

    alg = FedAvg(
        data_manager=dm,
        num_clients=n_clients,
        sample_scheme="uniform",
        sample_rate=0.01,
        model_def=partial(SimpleCNN2, num_channels=3),
        epochs=5,
        criterion_def=partial(CrossEntropyLoss, log_freq=100),
        batch_size=32,
        metric_logger=sw,
        device="cuda",
    )
    alg.hook_local_score(
        partial(Accuracy, log_freq=50),
        split='train,
        score_name="accuracy",
    )
    alg.hook_global_score(
        partial(Accuracy, log_freq=40),
        split='test,
        score_name="accuracy",
    )
    report_summary = alg.train(rounds=50)

Side Notes
==========
* Do not use double underscores (`__`) in argument names of your customized classes.
