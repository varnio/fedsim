Example
-------
.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter
    from fedsim.distributed.centralized.training import FedAvg
    from fedsim.distributed.data_management import BasicDataManager
    from fedsim.models.mcmahan_nets import cnn_cifar100
    from fedsim.scores import cross_entropy
    from fedsim.scores import accuracy


    n_clients = 1000

    dm = BasicDataManager("./data", "cifar100", n_clients)
    sw = SummaryWriter()

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
