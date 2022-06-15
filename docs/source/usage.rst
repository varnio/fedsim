Usage
=====

Package
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