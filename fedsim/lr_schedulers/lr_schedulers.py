r"""
Round to Round Learning Rate Schedulers
---------------------------------------

Used to schedule the initial learning rate of the local learning rate at each round

"""

import math

from torch._six import inf


class LRScheduler(object):
    def __init__(self, init_lr: float, verbose=False) -> None:
        self.base_lr = init_lr
        self._cur_lr = init_lr
        self._step_count = 0

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def get_lr(self):
        return self._cur_lr

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate."""
        if is_verbose:
            if epoch is None:
                print(
                    "Adjusting learning rate"
                    " of group {} to {:.4e}.".format(group, lr)
                )
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    "Epoch {}: adjusting learning rate"
                    " of group {} to {:.4e}.".format(epoch_str, group, lr)
                )


class StepLR(LRScheduler):
    def __init__(self, init_lr: float, step_size, gamma) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(init_lr)

    def step(self):
        self._step_count += 1
        if self._step_count % self.step_size == 0:
            self._cur_lr = self._cur_lr * self.gamma


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, init_lr: float, T_0, T_mult=1, eta_min=0, verbose=False) -> None:
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(CosineAnnealingWarmRestarts, self).__init__(init_lr)

    def step(self):
        self._step_count += 1
        if self._step_count >= self.T_0:
            if self.T_mult == 1:
                self.T_cur = self._step_count % self.T_0
            else:
                n = int(
                    math.log(
                        (self._step_count / self.T_0 * (self.T_mult - 1) + 1),
                        self.T_mult,
                    )
                )
                self.T_cur = self._step_count - self.T_0 * (self.T_mult**n - 1) / (
                    self.T_mult - 1
                )
                self.T_i = self.T_0 * self.T_mult ** (n)
        else:
            self.T_i = self.T_0
            self.T_cur = self._step_count
        self._cur_lr = (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
        )


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        init_lr (float): initial learning rate.
        trigger_metric (str): name of the metric to pass to step. This should be a
            hooked score, existing in the final report.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float): A scalar to lower bound the learning rate
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(0.1, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(
        self,
        init_lr,
        trigger_metric="clients.train_loss",
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        self.init_lr = init_lr
        self._cur_lr = init_lr
        self.trigger_metric = trigger_metric
        self.min_lr = min_lr

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = self._cur_lr

    def _reduce_lr(self, epoch):
        old_lr = self._cur_lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self._cur_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    "Epoch {}: reducing learning rate"
                    " to {:.4e}.".format(epoch_str, new_lr)
                )

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )

    def get_lr(self):
        return self._cur_lr


class StepLRWithRestartOnPlateau(ReduceLROnPlateau):
    def __init__(
        self,
        init_lr,
        step_size,
        trigger_metric="clients.train_loss",
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=10e-7,
        eps=1e-8,
        verbose=False,
    ):
        super(StepLRWithRestartOnPlateau, self).__init__(
            init_lr,
            trigger_metric,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            verbose,
        )
        self.step_size = step_size

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._restart_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        else:
            if self.num_bad_epochs % self.step_size == 0:
                self._cur_lr = self._cur_lr * self.factor

        self._last_lr = self._cur_lr

    def _restart_lr(self, epoch):
        self._cur_lr = self.init_lr
        if self.verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print(
                "Epoch {}: restarted learning rate"
                " to {:.4e}.".format(epoch_str, self._cur_lr)
            )


class CosineAnnealingWithRestartOnPlateau(ReduceLROnPlateau):
    def __init__(
        self,
        init_lr,
        T_0,
        trigger_metric="clients.train_loss",
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=10e-7,
        eps=1e-8,
        verbose=False,
    ):
        super().__init__(
            init_lr,
            trigger_metric,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            verbose,
        )
        self.T_0 = T_0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._restart_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        else:
            if self.num_bad_epochs < self.T_0:
                self._cur_lr = (
                    self.min_lr
                    + (self.init_lr - self.min_lr)
                    * (1 + math.cos(math.pi * self.num_bad_epochs / self.T_0))
                    / 2
                )

        self._last_lr = self._cur_lr

    def _restart_lr(self, epoch):
        self._cur_lr = self.init_lr
        if self.verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print(
                "Epoch {}: restarted learning rate"
                " to {:.4e}.".format(epoch_str, self._cur_lr)
            )
