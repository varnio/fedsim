Usage
=====

Package
---------

Here is a demo:

.. code-block:: python

   from fedsim.data_manager.basic_data_manager import BasicDataManager
   from fedsim.fl.algorithm import FedAvg
   from fedsim.model.mcmahan_nets import mlp_mnist

   n_clients = 500
   dataset_name = 'mnist'

   dm = BasicDataManager(root ='./data', dataset_name, n_client)
   sw = SummaryWriter()

   alg = FedAvg(
       data_manager=dm,
       num_clients=n_clients,
       sample_scheme='uniform',
       sample_rate=0.01,
       model_class=mlp_mnist,
       epochs=5,
       loss_fn='ce',
       metric_logger=sw,
       device='cuda',

   )

   alg.train(rounds=100)

Included cli tool
-----------------

For help with cli check here:

.. code-block:: bash

   fedsim --help