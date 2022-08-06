r"""
Round to Round Learning Rate Schedulers
---------------------------------------

Used to schedule the initial learning rate of the local learning rate at each round

"""

import math


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
