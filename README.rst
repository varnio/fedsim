FedSim
======

FedSim is a Generic Federated Learning Simulator. It aims to provide the researchers with an easy to develope/maintain simulator for Federated Learning. See documentation at `here <https://fedsim.readthedocs.io/en/main/>`_!

Installation
============

.. code-block:: bash

   pip install fedsim

Usage
=====

As module
---------

Here is a demo:

.. code-block:: python

    from fedsim.data_manager.basic_data_manager import BasicDataManager
    from fedsim.fl.algorithms.fedavg import FedAvg
    from fedsim.models.mcmahan_nets import cnn_cifar100
    from torch.utils.tensorboard import SummaryWriter

    n_clients = 500

    dm = BasicDataManager('./data', 'cifar100', n_clients)
    sw = SummaryWriter()

    alg = FedAvg(
        data_manager=dm,
        num_clients=n_clients,
        sample_scheme='uniform',
        sample_rate=0.01,
        model_class=cnn_cifar100,
        epochs=5,
        loss_fn='ce',
        batch_size=32,
        metric_logger=sw,
        device='cuda',

    )

    alg.train(rounds=5)

Included cli tool
-----------------

For help with cli check here:

.. code-block:: bash

   fedsim --help

DataManager
===========

Any custome DataManager class should inherit from ``fedsim.data_manager.data_manager.DataManager`` (or its children) and implement its abstract methods. For example:

.. code-block:: python

   from fedsim.data_manager.data_manager import DataManager

   class CustomDataManager(DataManager)
       def __init__(self, root, other_arg, ...):
           self.other_arg = other_arg
           # note that super should be called at the end of init \
           # because the abstract classes are called in its __init__
           super(BasicDataManager, self).__init__(root, seed, save_path=save_path)

       def make_datasets(self, root: str) -> Iterable[Dict[str, object]]:
           """Abstract method to be implemented by child class.

           Args:
               dataset_name (str): name of the dataset.
               root (str): directory to download and manipulate data.
               save_path (str): directory to store the data after partitioning.

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

Integration with included cli
-----------------------------

To automatically include your custom data manager in the provided cli tool, you can place your class in a file under ``fedsim/data_manager``. Then, call it using option ``--data-manager``. To deliver arguments to the ``__init__`` method of your custom data manager, you can pass options in form of ``--d-<arg-name>`` where ``<arg-name>`` is the argument. Example

.. code-block:: bash

   fedsim fed-learn --data-manager CustomDataManager --d-other_arg <other_arg_value> ...

Included DataManager
--------------------

Provided with the simulator is a basic DataManager called ``BasicDataManager`` which for now supports the following datasets


* `MNIST <http://yann.lecun.com/exdb/mnist/>`_
* `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
* `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_

It supports the popular partitioning schemes (iid, Dirichlet distribution, unbalanced, etc.).

FLAlgorithm
===========

Any custome DataManager class should inherit from ``fedsim.fl.fl_algorithm.FLAlgorithm`` (or its children) and implement its abstract methods. For example:

.. code-block:: python

   from typing import Optional, Hashable, Mapping, Dict, Any
   from fedsim.fl.fl_algorithm import FLAlgorithm

   class CustomFLAlgorithm(FLAlgorithm):
       def __init__(
           self, data_manager, num_clients, sample_scheme, sample_rate, model_class, epochs, loss_fn,
           batch_size, test_batch_size, local_weight_decay, slr, clr, clr_decay, clr_decay_type, 
           min_clr, clr_step_size, metric_logger, device, log_freq, other_arg, ... , *args, **kwargs,
       ):
           self.other_arg = other_arg

           super(FedAvg, self).__init__(
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

Integration with included cli
-----------------------------

To automatically include your custom algorithm by the provided cli tool, you can place your class in a file under fedsim/fl/algorithms. Then, call it using option --algorithm. To deliver arguments to the **init** method of your custom algorithm, you can pass options in form of `--a-<arg-name>` where `<arg-name>` is the argument. Example

.. code-block:: bash

   fedsim fed-learn --algorithm CustomFLAlgorithm --a-other_arg <other_arg_value> ...

other attributes and methods provide by FLAlgorithm
---------------------------------------------------

.. list-table::
   :header-rows: 1

   * - method
     - functionality
   * - ``FLAlgorithm.get_model_class()``
     - returns the class object of the model architecture
   * - ``FLAlgorithm.write_server(key, obj)``
     - stores obj in server memory, accessible with key
   * - ``FLAlgorithm.write_client(client_id, key, obj)``
     - stores obj in client_id's memory, accessible with key
   * - ``FLAlgorithm.read_server(key)``
     - returns obj associated with key in server memory
   * - ``FLAlgorithm.read_client(client_id, key)``
     - returns obj associated with key in client_id's memory


Included FL algorithms
----------------------

.. list-table::
   :header-rows: 1

   * - alias
     - paper
   * - fedavg
     - `Communication-Efficient Learning of Deep Networks from Decentralized Data <https://arxiv.org/abs/1602.05629>`_
   * - fedavg
     - `Federated Optimization in Heterogeneous Networks <https://arxiv.org/abs/1812.06127>`_
   * - fedavgm
     - `Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification <https://arxiv.org/abs/1909.06335>`_
   * - fednova
     - `Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization <https://arxiv.org/abs/2007.07481>`_
   * - fedprox
     - `Federated Optimization in Heterogeneous Networks <https://arxiv.org/abs/1812.06127>`_
   * - feddyn
     - `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_
   * - adabest
     - `Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation <https://arxiv.org/abs/2204.13170>`_


Model Architectures
===================

Included Architectures
----------------------

The models used by `FedAvg paper <https://arxiv.org/abs/1602.05629>`_ are supported:


* McMahan's 2 layer mlp for MNIST
* McMahan's CNN for CIFAR10 and CIFAR100

To use them import ``fedsim.model.mcmahan_nets``.

Integration with included cli
-----------------------------

If you want to use a custom pytorch class model with the cli tool, then you can simply place it under ``fedsim.models`` and call it:

.. code-block:: bash

   fedsim fed-learn --model CustomModule ...

Contributor's Guide
===================

Style
-----


* 
  We use ``yapf`` for formatting the style of the code. Before your merge request:


  * make sure ``yapf`` is installed.
  * inyour terminal, locate at the root of the project
  * launch the following command: ``yapf -ir -vv --no-local-style ./``

* 
  For now, type hinting is only used to avoid confusion at certain points.

TODO
====


* [ ] only make local test available when log-freq triggers
* [ ] add implementation of scaffold
* [ ] publish the code
* [ ] add doc (sphinx)
