r""" This file contains an implementation of the following paper:
    Title: "Measuring the Effects of Non-Identical Data Distribution for
    ---- Federated Visual Classification"
    Authors: Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown
    Publication date: September 13th, 2019
    Link: https://arxiv.org/abs/1909.06335
"""
from torch.optim import SGD

from . import fedavg


class FedAvgM(fedavg.FedAvg):
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
        momentum=0.9,
        *args,
        **kwargs,
    ):

        self.momentum = momentum

        super(FedAvgM, self).__init__(
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
        # over write optimizer
        params = self.read_server("cloud_params")
        optimizer = SGD(params=[params], lr=slr, momentum=self.momentum)
        self.write_server("optimizer", optimizer)
