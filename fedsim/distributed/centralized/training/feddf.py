r"""
AvgLogits
---------
"""

from torch.nn.functional import log_softmax
from torch.nn.utils.stateless import functional_call

from fedsim.scores import KLDivScore
from fedsim.utils import SerialAggregator
from fedsim.utils import initialize_module
from fedsim.utils import vector_to_named_parameters_like

from .fedavg import FedAvg
from .utils import serial_aggregation


class FedDF(FedAvg):
    r"""Ensemble Distillation for Robust Model Fusion in Federated Learning.

    For further details regarding the algorithm we refer to `Ensemble Distillation for
    Robust Model Fusion in Federated Learning`_.

    Args:
        data_manager (``distributed.data_management.DataManager``): data manager
        metric_logger (``logall.Logger``): metric logger for tracking.
        num_clients (int): number of clients
        sample_scheme (``str``): mode of sampling clients. Options are ``'uniform'``
            and ``'sequential'``
        sample_rate (``float``): rate of sampling clients
        model_def (``torch.Module``): definition of for constructing the model
        epochs (``int``): number of local epochs
        criterion_def (``Callable``): loss function defining local objective
        optimizer_def (``Callable``): derfintion of server optimizer
        local_optimizer_def (``Callable``): defintoin of local optimizer
        lr_scheduler_def (``Callable``): definition of lr scheduler of server optimizer.
        local_lr_scheduler_def (``Callable``): definition of lr scheduler of local
            optimizer
        r2r_local_lr_scheduler_def (``Callable``): definition to schedule lr that is
            delivered to the clients at each round (deterimined init lr of the
            client optimizer)
        batch_size (int): batch size of the local trianing
        test_batch_size (int): inference time batch size
        device (str): cpu, cuda, or gpu number
        global_train_split (str): the name of train split to be used on server
        global_epochs (int): number of training epochs on the server

    .. note::
        definition of
            * learning rate schedulers, could be any of the ones defined at
                ``torch.optim.lr_scheduler`` or any other that implements step and
                get_last_lr methods._schedulers``.
            * optimizers, could be any ``torch.optim.Optimizer``.
            * model, could be any ``torch.Module``.
            * criterion, could be any ``fedsim.scores.Score``.

    .. warning::
        this algorithm needs a split for training on the server. This means that the
        global datasets provided in data manager should include an extra split.


    .. _Ensemble Distillation for Robust Model Fusion in Federated Learning:
        https://openreview.net/forum?id=gjrMQoAhSRq
    """

    def init(server_storage, *args, **kwrag):
        default_global_train_split = "valid"
        default_global_epochs = 1
        FedAvg.init(server_storage)
        server_storage.write(
            "global_train_split",
            kwrag.get("global_train_split", default_global_train_split),
        )
        server_storage.write(
            "global_epochs",
            kwrag.get("global_epochs", default_global_epochs),
        )

    def receive_from_client(
        server_storage,
        client_id,
        client_msg,
        train_split_name,
        serial_aggregator,
        appendix_aggregator,
    ):
        params = client_msg["local_params"].clone().detach().data
        appendix_aggregator.append("local_params", params)
        return serial_aggregation(
            server_storage,
            client_id,
            client_msg,
            train_split_name,
            serial_aggregator,
        )

    def optimize(server_storage, serial_aggregator, appendix_aggregator):
        if "local_params" in serial_aggregator:
            param_avg = serial_aggregator.pop("local_params")
            cloud_params = server_storage.read("cloud_params")
            cloud_params.data = param_avg.data

            model = server_storage.read("model")
            global_train_split = server_storage.read("global_train_split")
            train_data_loader = server_storage.read("global_dataloaders").get(
                global_train_split
            )
            if train_data_loader is None:
                raise Exception(
                    f"no dataloader made for split {global_train_split} on the server!"
                )
            global_epochs = server_storage.read("global_epochs")
            optimizer = server_storage.read("optimizer")
            lr_scheduler = server_storage.read("lr_scheduler")
            device = server_storage.read("device")

            for _ in range(global_epochs):
                for x, _ in train_data_loader:
                    x = x.to(device)
                    target_agg = SerialAggregator()
                    for local_params in appendix_aggregator.get_values("local_params"):
                        initialize_module(model, local_params)
                        target = model(x).clone().detach()
                        target_agg.add("target", target, 1)

                    target_out = log_softmax(target_agg.get("target"), 1)

                    # initialize_module(model, cloud_params)
                    criterion = KLDivScore(log_target=True)
                    param_dict = vector_to_named_parameters_like(
                        cloud_params, model.named_parameters()
                    )
                    pred = functional_call(model, param_dict, x)
                    pred_out = log_softmax(pred, 1)

                    loss = criterion(pred_out, target_out.data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    del target_agg

        return serial_aggregator.pop_all()
