r""" This file contains an implementation of the following paper:
    Title: "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
    Authors: Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor
    Publication date: 15 Jul 2020
    Link: https://arxiv.org/abs/2007.07481
"""
from . import fedavg


class FedNova(fedavg.FedAvg):

    def __init__(
        self,
        data_manager,
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
        metric_logger,
        device,
        log_freq,
        *args,
        **kwargs,
    ):
        super(FedNova, self).__init__(
            data_manager,
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
            metric_logger,
            device,
            log_freq,
        )

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = client_msg['num_samples'] / client_msg['num_steps']
        self.agg(client_id, client_msg, aggregation_results, weight=weight)
