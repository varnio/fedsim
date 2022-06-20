r""" This file contains an implementation of the following paper:
    Title: "Tackling the Objective Inconsistency Problem in Heterogeneous
    ---- Federated Optimization"
    Authors: Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor
    Publication date: 15 Jul 2020
    Link: https://arxiv.org/abs/2007.07481
"""
from . import fedavg


class FedNova(fedavg.FedAvg):
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
        batch_size=32,
        test_batch_size=64,
        local_weight_decay=0.0,
        slr=1.0,
        clr=0.1,
        clr_decay=1.0,
        clr_decay_type="step",
        min_clr=1e-12,
        clr_step_size=1000,
        device="cuda",
        log_freq=10,
        *args,
        **kwargs,
    ):
        super(FedNova, self).__init__(
            data_manager,
            metric_logger,
            num_clients,
            sample_scheme,
            sample_rate,
            model_class,
            epochs,
            loss_fn,
            batch_size,
            test_batch_size,
            local_weight_decay,
            slr,
            clr,
            clr_decay,
            clr_decay_type,
            min_clr,
            clr_step_size,
            device,
            log_freq,
        )

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = client_msg["num_samples"] / client_msg["num_steps"]
        self.agg(client_id, client_msg, aggregation_results, weight=weight)
