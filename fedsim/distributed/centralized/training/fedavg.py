r"""
FedAvg
------
"""
import math
import sys
from copy import deepcopy
from functools import partial

from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from fedsim.local.training import local_inference
from fedsim.local.training import local_train
from fedsim.local.training.step_closures import default_step_closure

from ..centralized_fl_algorithm import CentralFLAlgorithm


class FedAvg(CentralFLAlgorithm):
    r"""Implements FedAvg algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Communication-Efficient
    Learning of Deep Networks from Decentralized Data`_.

    Args:
        data_manager (Callable): data manager
        metric_logger (Callable): a logall.Logger instance
        num_clients (int): number of clients
        sample_scheme (str): mode of sampling clients
        sample_rate (float): rate of sampling clients
        model_class (Callable): class for constructing the model
        epochs (int): number of local epochs
        loss_fn (Callable): loss function defining local objective
        optimizer_class (Callable): server optimizer class
        local_optimizer_class (Callable): local optimization class
        lr_scheduler_class: class definition for lr scheduler of server optimizer
        local_lr_scheduler_class: class definition for lr scheduler of local optimizer
        r2r_local_lr_scheduler_class: class definition to schedule lr delivered to
            clients at each round (init lr of the client optimizer)
        batch_size (int): local trianing batch size
        test_batch_size (int): inference time batch size
        device (str): cpu, cuda, or gpu number
        log_freq (int): frequency of logging

    .. _Communication-Efficient Learning of Deep Networks from Decentralized
        Data: https://arxiv.org/abs/1602.05629
    """

    def __init__(
        self,
        data_manager,
        metric_logger,
        num_clients,
        sample_scheme,
        sample_rate,
        model_class,
        epochs,
        loss_fn,
        optimizer_class=partial(SGD, lr=0.1, weight_decay=0.001),
        local_optimizer_class=partial(SGD, lr=1.0),
        lr_scheduler_class=None,
        local_lr_scheduler_class=None,
        r2r_local_lr_scheduler_class=None,
        batch_size=32,
        test_batch_size=64,
        device="cuda",
        log_freq=10,
    ):
        super(FedAvg, self).__init__(
            data_manager,
            metric_logger,
            num_clients,
            sample_scheme,
            sample_rate,
            model_class,
            epochs,
            loss_fn,
            optimizer_class,
            local_optimizer_class,
            lr_scheduler_class,
            local_lr_scheduler_class,
            r2r_local_lr_scheduler_class,
            batch_size,
            test_batch_size,
            device,
            log_freq,
        )

        # make mode and optimizer
        model = self.get_model_class()().to(self.device)
        params = deepcopy(parameters_to_vector(model.parameters()).clone().detach())
        optimizer = optimizer_class(params=[params])
        lr_scheduler = None
        if lr_scheduler_class is not None:
            lr_scheduler = lr_scheduler_class(optimizer)
        # write model and optimizer to server
        self.write_server("model", model)
        self.write_server("cloud_params", params)
        self.write_server("optimizer", optimizer)
        self.write_server("lr_scheduler", lr_scheduler)

    def send_to_client(self, client_id):
        # since fedavg broadcast the same model to all selected clients,
        # the argument client_id is not used

        # load cloud stuff
        cloud_params = self.read_server("cloud_params")
        model = self.read_server("model")

        # copy cloud params to cloud model to send to the client
        vector_to_parameters(cloud_params.detach().clone().data, model.parameters())
        # return a copy of the cloud model
        return dict(model=model)

    def send_to_server(
        self,
        client_id,
        datasets,
        epochs,
        loss_fn,
        batch_size,
        optimizer_class,
        lr_scheduler_class=None,
        device="cuda",
        ctx=None,
        step_closure=None,
        *args,
        **kwargs,
    ):
        # create a random sampler with replacement so that
        # stochasticity is maximiazed and privacy is not compromized
        sampler = RandomSampler(
            datasets["train"],
            replacement=True,
            num_samples=math.ceil(len(datasets["train"]) / batch_size) * batch_size,
        )
        # # create train data loader
        train_loader = DataLoader(
            datasets["train"], batch_size=batch_size, sampler=sampler
        )

        model = ctx["model"]
        optimizer = optimizer_class(model.parameters())
        lr_scheduler = None
        if lr_scheduler_class is not None:
            lr_scheduler = lr_scheduler_class(optimizer=optimizer)
        # optimize the model locally
        step_closure_ = default_step_closure if step_closure is None else step_closure
        opt_result = local_train(
            model,
            train_loader,
            epochs,
            0,
            loss_fn,
            optimizer,
            lr_scheduler,
            device,
            step_closure_,
            metric_fn_dict={
                f"train_{key}": score
                for key, score in self.get_local_score_functions("train").items()
            },
        )
        (
            num_train_samples,
            num_steps,
            diverged,
            loss,
            metrics,
        ) = opt_result
        # local test
        if "test" in datasets:
            test_loader = DataLoader(
                datasets["test"],
                batch_size=batch_size,
                shuffle=False,
            )
            test_metrics, num_test_samples = local_inference(
                model,
                test_loader,
                metric_fn_dict={
                    f"test_{key}": score
                    for key, score in self.get_local_score_functions("test").items()
                },
                device=device,
            )
        else:
            test_metrics = dict()
            num_test_samples = 0

        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
            num_steps=num_steps,
            diverged=diverged,
            train_loss=loss,
            metrics=metrics,
            num_test_samples=num_test_samples,
            test_metrics=test_metrics,
        )

    def agg(self, client_id, client_msg, aggregator, weight=1):
        params = client_msg["local_params"].clone().detach().data
        diverged = client_msg["diverged"]
        loss = client_msg["train_loss"]
        metrics = client_msg["metrics"]
        test_metrics = client_msg["test_metrics"]
        n_ts_samples = client_msg["num_test_samples"]

        if diverged:
            print("client {} diverged".format(client_id))
            print("exiting ...")
            sys.exit(1)

        aggregator.add("local_params", params, weight)
        aggregator.add("clients.train_loss", loss, weight)
        for key, metric in metrics.items():
            aggregator.add("clients.{}".format(key), metric, weight)
        for key, metric in test_metrics.items():
            aggregator.add("clients.{}".format(key), metric, n_ts_samples)

        # purge client info
        del client_msg

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = client_msg["num_samples"]
        if weight > 0:
            self.agg(client_id, client_msg, aggregation_results, weight=weight)

    def optimize(self, aggregator):
        if "local_params" in aggregator:
            param_avg = aggregator.pop("local_params")
            optimizer = self.read_server("optimizer")
            lr_scheduler = self.read_server("lr_scheduler")
            cloud_params = self.read_server("cloud_params")
            pseudo_grads = cloud_params.data - param_avg
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = pseudo_grads
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            # purge aggregated results
            del param_avg
        return aggregator.pop_all()

    def deploy(self):
        return dict(avg=self.read_server("cloud_params"))

    def report(
        self,
        dataloaders,
        metric_logger,
        device,
        optimize_reports,
        deployment_points=None,
    ):
        model = self.read_server("model")
        metrics_from_deployment = dict()
        if deployment_points is not None:
            for point_name, point in deployment_points.items():
                # copy cloud params to cloud model to send to the client
                vector_to_parameters(point.detach().clone().data, model.parameters())

                for split_name, loader in dataloaders.items():
                    metrics, _ = local_inference(
                        model,
                        loader,
                        metric_fn_dict={
                            f"server.{point_name}.{split_name}_{key}": score
                            for key, score in self.get_global_score_functions(
                                split_name
                            ).items()
                        },
                        device=device,
                    )
                    metrics_from_deployment = {**metrics_from_deployment, **metrics}
        return {**metrics_from_deployment, **optimize_reports}
