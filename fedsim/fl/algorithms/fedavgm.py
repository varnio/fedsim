r""" This file contains an implementation of the following paper:
    Title: "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification"
    Authors: Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown
    Publication date: September 13th, 2019
    Link: https://arxiv.org/abs/1909.06335
"""
from . import fedavg
from torch.optim import SGD


class FedAvgM(fedavg.FedAvg):

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
        momentum=0.9,
        *args,
        **kwargs,
    ):

        self.momentum = momentum

        super(FedAvgM, self).__init__(
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
        # over write optimizer
        params = self.read_server('cloud_params')
        optimizer = SGD(params=[params], lr=slr, momentum=self.momentum)
        self.write_server('optimizer', optimizer)
