r"""
Fedsim Losses
-------------
"""
import torch

from . import scores

Tensor = torch.Tensor


class CrossEntropyLoss(scores.CrossEntropyScore):
    r"""updatable cross entropy loss"""

    def __init__(
        self,
        split,
        loss_name="cross_entropy_loss",
        weight=None,
        label_smoothing=0.0,
    ) -> None:
        super().__init__(
            eval_freq=1,
            split=split,
            score_name=loss_name,
            reduction="micro",
            weight=weight,
            label_smoothing=label_smoothing,
        )

    def __call__(self, input, target) -> Tensor:
        r"""updates the cross entropy loss on a mini-batch detached from the
        computational graph. It also returns the current batch score without detaching
        from the graph.

        Args:

            input (Tensor) : Predicted unnormalized scores (often referred to as\
                logits); see Shape section below for supported shapes.
            target (Tensor) : Ground truth class indices or class probabilities;
                see Shape section below for supported shapes.

        Shape:

            - Input: shape :math:`(C)`, :math:`(N, C)`.
            - Target: shape :math:`()`, :math:`(N)` where each
                value should be between :math:`[0, C)`.

            where:

            .. math::
                \begin{aligned}
                    C ={} & \text{number of classes} \\
                    N ={} & \text{batch size} \\
                \end{aligned}

        Returns:
            Tensor: cross entropy loss of current batch

        """
        return super().__call__(input, target)
