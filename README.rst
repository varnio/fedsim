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

Usage
=====

As package
----------

Here is a demo:

.. code-block:: python

    from logall import TensorboardLogger
    from fedsim.distributed.centralized.training import FedAvg
    from fedsim.distributed.data_management import BasicDataManager
    from fedsim.models.mcmahan_nets import cnn_cifar100
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
        model_def=cnn_cifar100,
        epochs=5,
        criterion_def=partial(CrossEntropyLoss, log_freq=100),
        batch_size=32,
        metric_logger=sw,
        device="cuda",
    )
    alg.hook_local_score(
        partial(Accuracy, log_freq=50),
        split_name='train,
        score_name="accuracy",
    )
    alg.hook_global_score(
        partial(Accuracy, log_freq=40),
        split_name='test,
        score_name="accuracy",
    )
    report_summary = alg.train(rounds=1)


fedsim-cli tool
---------------

For help with cli check here:

.. code-block:: bash

   fedsim-cli --help

DataManager
===========

Any custom DataManager class should inherit from ``fedsim.data_manager.data_manager.DataManager`` (or its children) and implement its abstract methods. For example:

.. code-block:: python

   from fedsim.distributed.data_management import DataManager

   class CustomDataManager(DataManager)
       def __init__(self, root, other_arg, ...):
           self.other_arg = other_arg
           # note that super should be called at the end of init \
           # because the abstract classes are called in its __init__
           super(CustomDataManager, self).__init__(root, seed, save_dir=save_dir)

       def make_datasets(self, root: str) -> Iterable[Dict[str, object]]:
           """Abstract method to be implemented by child class.

           Args:
               dataset_name (str): name of the dataset.
               root (str): directory to download and manipulate data.
               save_dir (str): directory to store the data after partitioning.

           Raises:
               NotImplementedError: if the dataset_name is not defined

           Returns:
               Iterable[Dict[str, object]]: dict of local datasets [split:dataset]
                                            followed by global ones.
           """
           raise NotImplementedError


       def partition_local_data(self, datasets: Dict[str, object]) -> Dict[str, Iterable[Iterable[int]]]:
           raise NotImplementedError


       def get_identifiers(self) -> Sequence[str]:
           """ Returns identifiers
               to be used for saving the partition info.

           Raises:
               NotImplementedError: this abstract method should be
                   implemented by child classes

           Returns:
               Sequence[str]: a sequence of str identifing class instance
           """
           raise NotImplementedError

Integration with fedsim-cli (DataManager)
-----------------------------------------

To automatically include your custom data-manager by the provided cli tool, you can place your class in a python file and pass its path to `-d` or `--data-manager` option (without .py) followed by colon and name of the data-manager.
For example, if you have data-manager `DataManager` stored in `foo/bar/my_custom_dm.py`, you can pass `--data-manager foo/bar/my_custom_dm:DataManager`.

.. note::

    Arguments of the **init** method of any data-manager could be given in `arg:value` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --data-manager BasicDataManager num_clients:1100 ...

    .. code-block:: bash

        fedsim-cli fed-learn --data-manager foo/bar/my_custom_dm:DataManager arg1:value ...


Included DataManager
--------------------

Provided with the simulator is a basic DataManager called ``BasicDataManager`` which for now supports the following datasets


* `MNIST <http://yann.lecun.com/exdb/mnist/>`_
* `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
* `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_

It supports the popular partitioning schemes (iid, Dirichlet distribution, unbalanced, etc.).

CentralFLAlgorithm
==================

Any custom CentralFLAlgorithm class should inherit from ``fedsim.distributed.centralized.CentralFLAlgorithm`` (or its children) and implement its abstract methods. For example:

Architecture
------------

.. image:: ../_static/fedlearn.svg

Example
-------
.. code-block:: python

    from typing import Optional, Hashable, Mapping, Dict, Any
    from fedsim.distributed.centralized import CentralFLAlgorithm

    class CustomFLAlgorithm(CentralFLAlgorithm):
        def __init__(
            data_manager, metric_logger, num_clients, sample_scheme, sample_rate, model_def, epochs, criterion_def,
            optimizer_def, local_optimizer_def, lr_scheduler_def=None, local_lr_scheduler_def,
            r2r_local_lr_scheduler_def=None, batch_size=32, test_batch_size=64, device="cuda", other_arg, ...
        ):
            self.other_arg = other_arg
            ...

            super(CustomFLAlgorithm, self).__init__(
                data_manager, metric_logger, num_clients, sample_scheme, sample_rate, model_def, epochs, criterion_def,
                optimizer_def, local_optimizer_def, lr_scheduler_def=None, local_lr_scheduler_def,
                r2r_local_lr_scheduler_def=None, batch_size=32, test_batch_size=64, device="cuda",
            )
            # make mode and optimizer
            model = self.get_model_def()().to(self.device)
            params = deepcopy(parameters_to_vector(model.parameters()).clone().detach())
            optimizer = optimizer_def(params=[params])
            lr_scheduler = None
            if lr_scheduler_def is not None:
                lr_scheduler = lr_scheduler_def(optimizer=optimizer)
            # write model and optimizer to server
            self.write_server("model", model)
            self.write_server("cloud_params", params)
            self.write_server("optimizer", optimizer)
            self.write_server("lr_scheduler", lr_scheduler)
            ...

        def send_to_client(self, client_id: int) -> Mapping[Hashable, Any]:
            """ returns context to send to the client corresponding to the client_id.

            .. warning::
                Do not send shared objects like server model if you made any
                before you deepcopy it.

            Args:
                client_id (int): id of the receiving client

            Raises:
                NotImplementedError: abstract class to be implemented by child

            Returns:
                Mapping[Hashable, Any]: the context to be sent in form of a Mapping
            """
            ...

        def send_to_server(self, client_id: int, datasets: Dict[str, Iterable],
            round_scores: Dict[str, Dict[str, fedsim.scores.Score]], epochs: int, criterion: nn.Module,
            train_batch_size: int, inference_batch_size: int, optimizer_def: Callable,
            lr_scheduler_def: Optional[Callable] = None, device: Union[int, str] = "cuda",
            ctx: Optional[Dict[Hashable, Any]] = None) -> Mapping[str, Any]:
            """client operation on the recieved information.

            Args:
                client_id (int): id of the client
                datasets (Dict[str, Iterable]): this comes from Data Manager
                round_scores (Dict[str, Dict[str, fedsim.scores.Score]]): dictionary of
                    form {'split_name':{'score_name': score_def}} for global scores to
                    evaluate at the current round.
                epochs (``int``): number of epochs to train
                criterion (nn.Module): either 'ce' (for cross-entropy) or 'mse'
                train_batch_size (int): training batch_size
                inference_batch_size (int): inference batch_size
                optimizer_def (float): class for constructing the local optimizer
                lr_scheduler_def (float): class for constructing the local lr scheduler
                device (Union[int, str], optional): Defaults to 'cuda'.
                ctx (Optional[Dict[Hashable, Any]], optional): context reveived.

            Returns:
                Mapping[str, Any]: client context to be sent to the server
            """
            ...


        def receive_from_client(self, client_id: int, client_msg: Mapping[Hashable, Any], aggregator: Any):
            """ receive and aggregate info from selected clients

            Args:
                client_id (int): id of the sender (client)
                client_msg (Mapping[Hashable, Any]): client context that is sent
                aggregator (Any): aggregator instance to collect info

            """
            raise NotImplementedError

        def optimize(self, aggregator: Any) -> Mapping[Hashable, Any]:
            """ optimize server mdoel(s) and return metrics to be reported

            Args:
                aggregator (Any): Aggregator instance

            Returns:
                Mapping[Hashable, Any]: context to be reported
            """
            ...

        def deploy(self) -> Optional[Mapping[Hashable, Any]]:
            """ return Mapping of name -> parameters_set to test the model

            """
            raise NotImplementedError

        def report(self, dataloaders, round_scores: Dict[str, Dict[str, Any]], metric_logger: Any,
            device: str, optimize_reports: Mapping[Hashable, Any],
            deployment_points: Optional[Mapping[Hashable, torch.Tensor]] = None) -> None:
            """test on global data and report info

            Args:
                dataloaders (Any): dict of data loaders to test the global model(s)
                metric_logger (Any): the logging object (e.g., SummaryWriter)
                device (str): 'cuda', 'cpu' or gpu number
                optimize_reports (Mapping[Hashable, Any]): dict returned by optimzier
                deployment_points (Mapping[Hashable, torch.Tensor], optional): output of deploy method

            """
            ...


Integration with fedsim-cli (CentralFLAlgorithm)
------------------------------------------------

To automatically include your custom algorithm by the provided cli tool, you can place your class in a python and pass its path to `-a` or `--algorithm` option (without .py) followed by colon and name of the algorithm.
For example, if you have algorithm `CustomFLAlgorithm` stored in a `foo/bar/my_custom_alg.py`, you can pass `--algorithm foo/bar/my_custom_alg:CustomFLAlgorithm`.

.. note::

    Arguments of the **init** method of any algoritthm could be given in `arg:value` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --algorithm AdaBest mu:0.01 beta:0.6 ...

    .. code-block:: bash

        fedsim-cli fed-learn --algorithm foo/bar/my_custom_alg:CustomFLAlgorithm mu:0.01 ...


other attributes and methods provide by CentralFLAlgorithm
----------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - method
     - functionality
   * - ``CentralFLAlgorithm.get_model_def()``
     - returns the class object of the model architecture
   * - ``CentralFLAlgorithm.write_server(key, obj)``
     - stores obj in server memory, accessible with key
   * - ``CentralFLAlgorithm.write_client(client_id, key, obj)``
     - stores obj in client_id's memory, accessible with key
   * - ``CentralFLAlgorithm.read_server(key)``
     - returns obj associated with key in server memory
   * - ``CentralFLAlgorithm.read_client(client_id, key)``
     - returns obj associated with key in client_id's memory


Included FL algorithms
----------------------

.. list-table::
   :header-rows: 1

   * - Alias
     - Paper
   * - FedAvg
     - .. image:: https://img.shields.io/badge/arXiv-1602.05629-b31b1b.svg?style=flat-square
        :target: https://arxiv.org/abs/1602.05629
        :alt: arXiv

   * - FedNova
     - .. image:: https://img.shields.io/badge/arXiv-2007.07481-b31b1b.svg?style=flat-square
        :target: https://arxiv.org/abs/2007.07481
        :alt: arXiv

   * - FedProx
     - .. image:: https://img.shields.io/badge/arXiv-1812.06127-b31b1b.svg?style=flat-square
        :target: https://arxiv.org/abs/1812.06127
        :alt: arXiv

   * - FedDyn
     - .. image:: https://img.shields.io/badge/arXiv-2111.04263-b31b1b.svg?style=flat-square
        :target: https://arxiv.org/abs/2111.04263
        :alt: arXiv

   * - AdaBest
     - .. image:: https://img.shields.io/badge/arXiv-2204.13170-b31b1b.svg?style=flat-square
        :target: https://arxiv.org/abs/2204.13170
        :alt: arXiv


Model Architectures
===================

Included Architectures
----------------------

The models used by `FedAvg paper <https://arxiv.org/abs/1602.05629>`_ are supported:


* McMahan's 2 layer mlp for MNIST
* McMahan's CNN for CIFAR10 and CIFAR100

To use them import ``fedsim.model.mcmahan_nets``.

Integration with fedsim-cli
---------------------------

To automatically include your custom model by the provided cli tool, you can place your class in a python and pass its path to `-m` or `--model` option (without .py) followed by colon and name of the model.
For example, if you have model `CustomModel` stored in a `foo/bar/my_custom_model.py`, you can pass `--model foo/bar/my_custom_alg:CustomModel`.

.. note::

    Arguments of the **init** method of any model could be given in `arg:value` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --model cnn_mnist num_classes:8 ...

    .. code-block:: bash

        fedsim-cli fed-learn --model foo/bar/my_custom_alg:CustomModel num_classes:8 ...


Learning Rate Schedulers
========================

`fedsim-cli fed-learn` accepts 3 scheduler objects.

* **lr-scheduler:** learning rate scheduler for server optimizer.
* **local-lr-scheduler:** learning rate scheduler for client optimizer.
* **r2r-local-lr-scheduler:** schedules the initial learning rate that is delivered to the clients of each round.

These arguments are passed to instances of the centralized FL algorithms.

.. note::
    Choose learning rate schedulers from ``fedsim.lr_schedulers`` documented at `Lr Schedulers Page`_.

.. _Lr Schedulers Page: https://fedsim.varnio.com/en/latest/reference/fedsim.lr_schedulers.html



fedsim-cli examples
===================
The following command splits CIFAR100 on 1000 idd partitions and then uses AdaBest algorithm with :math:`\mu=0.02` and :math:`\beta=0.96` to train a model.
It randomly draws 1\% of all clients (200 clietns, first 200 paritions of the 1000) at each round (2 clients) and uses SGD with lr=0.05 and weight_decay=0.001 as for the local learning rate.
Local training batch size is 50.


.. code-block:: bash

    fedsim-cli fed-learn -a AdaBest mu:0.02 beta:0.96 -m cnn_cifar100 -d BasicDataManager dataset:cifar100 num_partitions:1000 -r 1001 -n 200 --local-optimizer SGD lr:0.05 weight_decay:0.001 --batch-size 50 --client-sample-rate 0.01

The following command tunes :math:`\mu` and :math:`\beta` for AdaBest algorithm. It uses Gaussian Process to maximize the average of the last 10 reported test accuracy scores.
:math:`\mu` is tuned for float numbers (Real) between 0 and 0.1 and :math:`\beta` is tuned for float numbers between 0.1 and 1. Notice that only 2 clients are defined while the data manager by default is splitting the data over 500 partitions.

.. code-block:: bash

    fedsim-cli fed-tune --epochs 1 --n-clients 2 --client-sample-rate 0.5 -a AdaBest mu:Real:0-0.1 beta:Real:0.3-1 --maximize-metric --n-iters 20

.. note::
    * To define a float range to tune use `Real` keyword as the argument value (e.g., `mu:Real:0-0.1`)
    * To define an integer range to tune use `Integer` keyword as the argument value (e.g., `arg1:Integer:2-15`)
    * To define a categorical range to tune use `Categorical` keyword as the argument value (e.g., `arg2:Categorical:uniform-normal-special`)

In the following command, CIFAR100 is split over 1000 partitions from which 100 are used in the FL setup. From those 100, 20 clietns are selected at random at each round for training.
The partitioning setup is non-iid with Dirichlet distribution factor :math:`\alpha=0.03`. The model architecture is cnn_cifar100.
Training goes for 10000 rounds and at each round initial local learning rate is determined by CosineAnnealing with period of 10 report points (which is equal to 500 rounds when reports are stored each 50 rounds as default).
The patience for `CosineAnnealingWithRestartOnPlateau` is set to 5 report points (250 rounds). In case patience is not violated at any point, learning rate is restarted to the initial values.

.. code-block:: bash

    fedsim-cli fed-learn -d BasicDataManager num_partitions:1000 seed:0 dataset:cifar100 rule:dir label_balance:0.03 -m cnn_cifar100 --rounds 10000 -n 100 --client-sample-rate 0.2 --r2r-local-lr-scheduler CosineAnnealingWithRestartOnPlateau verbose:True T_0:10 patience:5

Side Notes
==========
* Do not use double underscores (`__`) in argument names of your customized classes.
