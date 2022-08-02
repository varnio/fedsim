r"""
FedAvgM
-------
"""

from torch.optim import SGD

from . import fedavg


class FedAvgM(fedavg.FedAvg):
    r"""Implements FedAvgM algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Measuring the Effects of
    Non-Identical Data Distribution for Federated Visual Classification`_.

    Args:
        data_manager (Callable): data manager
        metric_logger (Callable): a logger object
        num_clients (int): number of clients
        sample_scheme (str): mode of sampling clients
        sample_rate (float): rate of sampling clients
        model_class (Callable): class for constructing the model
        epochs (int): number of local epochs
        loss_fn (Callable): loss function defining local objective
        batch_size (int): local trianing batch size
        test_batch_size (int): inference time batch size
        local_weight_decay (float): weight decay for local optimization
        slr (float): server learning rate
        clr (float): client learning rate
        clr_decay (float): round to round decay for clr (multiplicative)
        clr_decay_type (str): type of decay for clr (step or cosine)
        min_clr (float): minimum client learning rate
        clr_step_size (int): frequency of applying clr_decay
        device (str): cpu, cuda, or gpu number
        log_freq (int): frequency of logging
        momentum (float): momentum for server steps

    .. _Measuring the Effects of Non-Identical Data Distribution for Federated Visual
        Classification: https://arxiv.org/abs/1909.06335
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
