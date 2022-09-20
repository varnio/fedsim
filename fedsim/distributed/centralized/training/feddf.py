r"""
AvgLogits
---------
"""

from functools import partial

from torch.nn.functional import log_softmax
from torch.nn.utils.stateless import functional_call
from torch.optim import SGD

from fedsim.scores import KLDivScore
from fedsim.utils import SerialAggregator
from fedsim.utils import initialize_module
from fedsim.utils import vector_to_named_parameters_like
from fedsim.utils import vectorize_module_grads

from . import fedavg


class FedDF(fedavg.FedAvg):
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

    def __init__(
        self,
        data_manager,
        metric_logger,
        num_clients,
        sample_scheme,
        sample_rate,
        model_def,
        epochs,
        criterion_def,
        optimizer_def=partial(SGD, lr=0.1),
        local_optimizer_def=partial(SGD, lr=0.1),
        lr_scheduler_def=None,
        local_lr_scheduler_def=None,
        r2r_local_lr_scheduler_def=None,
        batch_size=32,
        test_batch_size=64,
        device="cuda",
        global_train_split="valid",
        global_epochs=1,
    ):
        super().__init__(
            data_manager,
            metric_logger,
            num_clients,
            sample_scheme,
            sample_rate,
            model_def,
            epochs,
            criterion_def,
            optimizer_def,
            local_optimizer_def,
            lr_scheduler_def,
            local_lr_scheduler_def,
            r2r_local_lr_scheduler_def,
            batch_size,
            test_batch_size,
            device,
        )
        server_storage = self.get_server_storage()
        server_storage.write("global_train_split", global_train_split)
        server_storage.write("global_epochs", global_epochs)

    def receive_from_client(
        self,
        server_storage,
        client_id,
        client_msg,
        train_split_name,
        aggregation_results,
    ):
        params = client_msg["local_params"].clone().detach().data
        diverged = client_msg["diverged"]
        metrics = client_msg["metrics"]
        n_samples = client_msg["num_samples"]

        if diverged:
            return False

        # logic_collector = list()
        if n_samples[train_split_name] > 0:
            # global_train_split = server_storage.read("global_train_split")
            # train_data_loader = self.get_global_loader_split(global_train_split)
            # model = server_storage.read("model")
            # load cloud params in model
            # cloud_params = server_storage.read("cloud_params")
            # initialize_module(model, cloud_params)
            # set model to eval
            # model.eval()

            aggregation_results.add(f"local_params.{client_id}", params, 1)
            ids = server_storage.read("ids")
            if ids is None:
                ids = [
                    client_id,
                ]
                server_storage.write("ids", ids)
            else:
                ids.append(client_id)

            for split_name, metrics in metrics.items():
                for key, metric in metrics.items():
                    aggregation_results.add(
                        f"clients.{split_name}.{key}", metric, n_samples[split_name]
                    )
        # purge client info
        del client_msg
        return True

    def optimize(self, server_storage, aggregator):
        ids = server_storage.read("ids")
        server_storage.write("ids", None)
        model = server_storage.read("model")
        cloud_params = server_storage.read("cloud_params")
        initialize_module(model, cloud_params)
        global_train_split = server_storage.read("global_train_split")
        train_data_loader = self.get_global_loader_split(global_train_split)
        global_epochs = server_storage.read("global_epochs")
        optimizer = server_storage.read("optimizer")
        lr_scheduler = server_storage.read("lr_scheduler")
        device = self.get_device()
        criterion = KLDivScore(log_target=True)
        for _ in range(global_epochs):
            for x, _ in train_data_loader:
                x = x.to(device)
                target_agg = SerialAggregator()
                for id in ids:
                    local_params = aggregator.get(f"local_params.{id}")
                    named_local_params = vector_to_named_parameters_like(
                        local_params, model.named_parameters()
                    )
                    target = functional_call(model, named_local_params, x).detach().data
                    target_agg.add("target", target, 1)

                target_out = log_softmax(target_agg.get("target"), 1)
                pred = model(x)
                pred_out = log_softmax(pred, 1)

                loss = criterion(pred_out, target_out.data)
                optimizer.zero_grad()
                loss.backward()

                grads = vectorize_module_grads(model)
                cloud_params.grad = grads

                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                del target_agg
                # update cloud parameters

        for id in ids:
            aggregator.pop(f"local_params.{id}")

        return aggregator.pop_all()
