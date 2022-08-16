r"""
Fedsim Scores
-------------
"""
import torch

Tensor = torch.Tensor


class Score(object):
    r"""Score base class.

    .. automethod:: __call__

    Args:
        log_freq (int, optional): how many steps gap between two evaluations.
            Defaults to 1.
        split (str, optional): data split to evaluate on . Defaults to 'test'.
        score_name (str): name of the score object
        reduction (str): Specifies the reduction to apply to the output:
            ``'micro'`` | ``'macro'``. ``'micro'``: as if mini-batches are
            concatenated. ``'macro'``: mean of score of each mini-batch
            (update). Default: ``'micro'``
    """

    def __init__(
        self,
        log_freq: int = 1,
        split="test",
        score_name="",
        reduction="micro",
    ) -> None:
        if log_freq < 1:
            raise Exception(f"invalid log_freq ({log_freq}) given to {score_name}")
        self.log_freq = log_freq
        self.split = split
        self.score_name = score_name
        self.reduction = reduction

    def get_name(self) -> str:
        r"""gives the name of the score

        Returns:
            str: score name
        """
        return self.score_name

    def __call__(self, input, target):
        r"""updates the score based on a mini-batch of input and target

        Args:
            input (Tensor) : Predicted unnormalized scores (often referred to as\
                logits); see Shape section below for supported shapes.
            target (Tensor) : Ground truth class indices or class probabilities;
                see Shape section below for supported shapes.

        Raises:
            NotImplementedError: This abstract method should be implemented by child
                classes
        """
        raise NotImplementedError

    def get_score(self) -> float:
        r"""returns the score

        Raises:
            NotImplementedError: This abstract method should be implemented by child
                classes

        Returns:
            float: the score
        """
        raise NotImplementedError

    def reset(self) -> None:
        """resets the internal buffers, makes it ready to start collecting

        Raises:
            NotImplementedError: This abstract method should be implemented by child
                classes
        """
        raise NotImplementedError


class Accuracy(Score):
    r"""updatable accuracy score

    .. automethod:: __call__

    Args:
        log_freq (int, optional): how many steps gap between two evaluations.
            Defaults to 1.
        split (str, optional): data split to evaluate on . Defaults to 'test'.
        score_name (str): name of the score object
        reduction (str): Specifies the reduction to apply to the output:
            ``'micro'`` | ``'macro'``. ``'micro'``: as if mini-batches are
            concatenated. ``'macro'``: mean of accuracy of each mini-batch
            (update). Default: ``'micro'``
    """

    def __init__(
        self,
        log_freq: int = 1,
        split="test",
        score_name="accuracy",
        reduction: str = "micro",
    ) -> None:
        super().__init__(log_freq, split, score_name, reduction)
        self._sum = 0
        self._weight = 0

    def __call__(self, input, target) -> Tensor:
        r"""updates the accuracy score on a mini-batch detached from the computational
        graph. It also returns the current batch score without detaching from the graph.

        Args:
            input (Tensor) : Predicted unnormalized scores (often referred to as\
                logits); see Shape section below for supported shapes.
            target (Tensor) : Ground truth class indices or class probabilities;
                see Shape section below for supported shapes.

        Shape:

            - Input: Shape :math:`(N, C)`.
            - Target: shape :math:`(N)` where each
                value should be between :math:`[0, C)`.

            where:

            .. math::
                \begin{aligned}
                    C ={} & \text{number of classes} \\
                    N ={} & \text{batch size} \\
                \end{aligned}

        Returns:
            Tensor: accuracy score of current batch


        """
        if not (target.shape[0] == input.shape[0]):
            raise Exception(
                f"size mismatch between input {input.shape[0]} and\
                {target.shape[0]}"
            )
        cur_sum = (input.argmax(dim=1) == target).sum()
        if self.reduction == "micro":
            self._sum += cur_sum.item()
            self._weight += input.shape[0]
        elif self.reduction == "macro":
            self._sum += cur_sum.item() / input.shape[0]
            self._weight += 1
        return cur_sum / input.shape[0]

    def get_score(self) -> float:
        if self._weight < 1:
            return 0
        return self._sum / self._weight

    def reset(self) -> None:
        self._sum = 0
        self._weight = 0


class CrossEntropyScore(Score):
    r"""updatable cross entropy score

    .. automethod:: __call__

    Args:
        log_freq (int, optional): how many steps gap between two evaluations.
            Defaults to 1.
        split (str, optional): data split to evaluate on . Defaults to 'test'.
        score_name (str): name of the score object
        reduction (str): Specifies the reduction to apply to the output:
        ``'micro'`` | ``'macro'``. ``'micro'``: as if mini-batches are
        concatenated. ``'macro'``: mean of cross entropy of each mini-batch
        (update). Default: ``'micro'``
    """

    def __init__(
        self,
        log_freq: int = 1,
        split="test",
        score_name="cross_entropy_score",
        weight=None,
        reduction: str = "micro",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(log_freq, split, score_name, reduction)
        self._base_class = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, reduction="sum"
        )
        self._sum = 0
        self._weight = 0

    def __call__(self, input, target) -> Tensor:
        r"""updates the cross entropy score on a mini-batch detached from the
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
            Tensor: cross entropy score of current batch


        """
        if not (target.shape[0] == input.shape[0]):
            raise Exception(
                f"size mismatch between input {input.shape[0]} and" f"{target.shape[0]}"
            )
        cur_sum = self._base_class(input, target)

        if self.reduction == "micro":
            self._sum += cur_sum.item()
            self._weight += input.shape[0]
        elif self.reduction == "macro":
            self._sum += cur_sum.item() / input.shape[0]
            self._weight += 1
        return cur_sum / input.shape[0]

    def get_score(self) -> float:
        if self._weight < 1:
            return 0
        return self._sum / self._weight

    def reset(self) -> None:
        self._sum = 0
        self._weight = 0
