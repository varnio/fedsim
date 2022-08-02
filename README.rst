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


FedSim is a Generic Federated Learning Simulator. It aims to provide the researchers with an easy to develope/maintain simulator for Federated Learning.
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
    from fedsim.scores import cross_entropy
    from fedsim.scores import accuracy


    n_clients = 1000

    dm = BasicDataManager("./data", "cifar100", n_clients)
    sw = TensorboardLogger(path=None)

    alg = FedAvg(
        data_manager=dm,
        num_clients=n_clients,
        sample_scheme="uniform",
        sample_rate=0.01,
        model_class=cnn_cifar100,
        epochs=5,
        loss_fn=cross_entropy,
        batch_size=32,
        metric_logger=sw,
        device="cuda",
    )
    alg.hook_global_score_function("test", "accuracy", accuracy)
    for key in dm.get_local_splits_names():
        alg.hook_local_score_function(key, "accuracy", accuracy)

    alg.train(rounds=1)


fedsim-cli tool
---------------

For help with cli check here:

.. code-block:: bash

   fedsim-cli --help

DataManager
===========

Any custome DataManager class should inherit from ``fedsim.data_manager.data_manager.DataManager`` (or its children) and implement its abstract methods. For example:

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

To automatically include your custom data-manager by the provided cli tool, you can place your class in a python file and pass its path to `-a` or `--data-manager` option (without .py) followed by column and name of the data-manager.
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

Any custome DataManager class should inherit from ``fedsim.distributed.centralized.CentralFLAlgorithm`` (or its children) and implement its abstract methods. For example:

.. code-block:: python

   from typing import Optional, Hashable, Mapping, Dict, Any
   from fedsim.distributed.centralized import CentralFLAlgorithm

   class CustomFLAlgorithm(CentralFLAlgorithm):
       def __init__(
           self, data_manager, num_clients, sample_scheme, sample_rate, model_class, epochs, loss_fn,
           batch_size, test_batch_size, local_weight_decay, slr, clr, clr_decay, clr_decay_type,
           min_clr, clr_step_size, metric_logger, device, log_freq, other_arg, ... , *args, **kwargs,
       ):
           self.other_arg = other_arg

           super(CustomFLAlgorithm, self).__init__(
               data_manager, num_clients, sample_scheme, sample_rate, model_class, epochs, loss_fn,
               batch_size, test_batch_size, local_weight_decay, slr, clr, clr_decay, clr_decay_type,
               min_clr, clr_step_size, metric_logger, device, log_freq,
           )
           # make mode and optimizer
           model = self.get_model_class()().to(self.device)
           params = deepcopy(
               parameters_to_vector(model.parameters()).clone().detach())
           optimizer = SGD(params=[params], lr=slr)
           # write model and optimizer to server
           self.write_server('model', model)
           self.write_server('cloud_params', params)
           self.write_server('optimizer', optimizer)
           ...

       def send_to_client(self, client_id: int) -> Mapping[Hashable, Any]:
           """ returns context to send to the client corresponding to client_id.
               Do not send shared objects like server model if you made any
               before you deepcopy it.

           Args:
               client_id (int): id of the receiving client

           Raises:
               NotImplementedError: abstract class to be implemented by child

           Returns:
               Mapping[Hashable, Any]: the context to be sent in form of a Mapping
           """
           raise NotImplementedError

       def send_to_server(
           self, client_id: int, datasets: Dict[str, Iterable], epochs: int, loss_fn: nn.Module,
           batch_size: int, lr: float, weight_decay: float = 0, device: Union[int, str] = 'cuda',
           ctx: Optional[Dict[Hashable, Any]] = None, *args, **kwargs
       ) -> Mapping[str, Any]:
           """ client operation on the recieved information.

           Args:
               client_id (int): id of the client
               datasets (Dict[str, Iterable]): this comes from Data Manager
               epochs (int): number of epochs to train
               loss_fn (nn.Module): either 'ce' (for cross-entropy) or 'mse'
               batch_size (int): training batch_size
               lr (float): client learning rate
               weight_decay (float, optional): weight decay for SGD. Defaults to 0.
               device (Union[int, str], optional): Defaults to 'cuda'.
               ctx (Optional[Dict[Hashable, Any]], optional): context reveived from server. Defaults to None.

           Raises:
               NotImplementedError: abstract class to be implemented by child

           Returns:
               Mapping[str, Any]: client context to be sent to the server
           """
           raise NotImplementedError

       def receive_from_client(self, client_id: int, client_msg: Mapping[Hashable, Any], aggregator: Any):
           """ receive and aggregate info from selected clients

           Args:
               client_id (int): id of the sender (client)
               client_msg (Mapping[Hashable, Any]): client context that is sent
               aggregator (Any): aggregator instance to collect info

           Raises:
               NotImplementedError: abstract class to be implemented by child
           """
           raise NotImplementedError

       def optimize(self, aggregator: Any) -> Mapping[Hashable, Any]:
           """ optimize server mdoel(s) and return metrics to be reported

           Args:
               aggregator (Any): Aggregator instance

           Raises:
               NotImplementedError: abstract class to be implemented by child

           Returns:
               Mapping[Hashable, Any]: context to be reported
           """
           raise NotImplementedError

       def deploy(self) -> Optional[Mapping[Hashable, Any]]:
           """ return Mapping of name -> parameters_set to test the model

           Raises:
               NotImplementedError: abstract class to be implemented by child
           """
           raise NotImplementedError

       def report(
           self, dataloaders, metric_logger: Any, device: str, optimize_reports: Mapping[Hashable, Any],
           deployment_points: Optional[Mapping[Hashable, torch.Tensor]] = None
       ) -> None:
           """test on global data and report info

           Args:
               dataloaders (Any): dict of data loaders to test the global model(s)
               metric_logger (Any): the logging object (e.g., SummaryWriter)
               device (str): 'cuda', 'cpu' or gpu number
               optimize_reports (Mapping[Hashable, Any]): dict returned by optimzier
               deployment_points (Mapping[Hashable, torch.Tensor], optional): output of deploy method

           Raises:
               NotImplementedError: abstract class to be implemented by child
           """
           raise NotImplementedError

Integration with fedsim-cli (CentralFLAlgorithm)
------------------------------------------------

To automatically include your custom algorithm by the provided cli tool, you can place your class in a python and pass its path to `-a` or `--algorithm` option (without .py) followed by column and name of the algorithm.
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
   * - ``CentralFLAlgorithm.get_model_class()``
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

   * - FedAvgM
     - .. image:: https://img.shields.io/badge/arXiv-1909.06335-b31b1b.svg?style=flat-square
        :target: https://arxiv.org/abs/1909.06335
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

To automatically include your custom model by the provided cli tool, you can place your class in a python and pass its path to `-m` or `--model` option (without .py) followed by column and name of the model.
For example, if you have model `CustomModel` stored in a `foo/bar/my_custom_model.py`, you can pass `--model foo/bar/my_custom_alg:CustomModel`.

.. note::

    Arguments of the **init** method of any model could be given in `arg:value` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --model cnn_mnist num_classes:8 ...

    .. code-block:: bash

        fedsim-cli fed-learn --model foo/bar/my_custom_alg:CustomModel num_classes:8 ...
