from fedsim.federation.algorithms import fedavg
from torch.optim import SGD


class Algorithm(fedavg.Algorithm):

    def __init__(
        self,
        data_manager,
        num_clients,
        sample_scheme,
        sample_rate,
        model,
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
        algorithm_params,
        metric_logger,
        device,
        log_freq,
        verbosity,
    ):
        super(Algorithm, self).__init__(
            data_manager,
            num_clients,
            sample_scheme,
            sample_rate,
            model,
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
            algorithm_params,
            metric_logger,
            device,
            log_freq,
            verbosity,
        )
        # over write optimizer
        params = self.read_server('cloud_params')
        optimizer = SGD(params=[params], lr=slr, momentum=self.momentum)
        self.write_server('optimizer', optimizer)

    def assign_default_params(self):
        return dict(momentum=0.9)
