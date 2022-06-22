from functools import partial

import torch
from sklearn import metrics as sk_metrics
from torch.nn import functional as F


def _torch_loss(y_true, y_pred, loss_name, normalize=True):
    """wrap torch loss functions with the order of arguments similar to scikit

    Args:
        y_true (List[float]): targets
        y_pred (List[float]): predictions
        loss_name (float): a loss fn defined in torch.nn.functional

    Returns:
        float: the value of the loss
    """
    reduction = "mean" if normalize else "sum"
    loss_fn = getattr(F, loss_name)
    return loss_fn(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        reduction=reduction,
    ).item()


accuracy = sk_metrics.accuracy_score
balanced_accuracy = sk_metrics.balanced_accuracy_score
top_k_accuracy = sk_metrics.top_k_accuracy_score
average_precision = sk_metrics.average_precision_score
neg_brier_score = sk_metrics.brier_score_loss
f1_micro = partial(sk_metrics.f1_score, average="micro")
f1_macro = partial(sk_metrics.f1_score, average="macro")
f1_weighted = partial(sk_metrics.f1_score, average="weighted")
f1_samples = partial(sk_metrics.f1_score, average="samples")
neg_log_loss = (sk_metrics.log_loss,)

precision_micro = partial(sk_metrics.precision_score, average="micro")
precision_macro = partial(sk_metrics.precision_score, average="macro")
precision_weighted = partial(sk_metrics.precision_score, average="weighted")
precision_samples = partial(sk_metrics.precision_score, average="samples")

recall_micro = partial(sk_metrics.recall_score, average="micro")
recall_macro = partial(sk_metrics.recall_score, average="macro")
recall_weighted = partial(sk_metrics.recall_score, average="weighted")
recall_samples = partial(sk_metrics.recall_score, average="samples")

jaccard_micro = partial(sk_metrics.jaccard_score, average="micro")
jaccard_macro = partial(sk_metrics.jaccard_score, average="macro")
jaccard_weighted = partial(sk_metrics.jaccard_score, average="weighted")
jaccard_samples = partial(sk_metrics.jaccard_score, average="samples")

# roc_auc for binary classification,
roc_auc_micro = partial(sk_metrics.roc_auc_score, average="micro")
roc_auc_macro = partial(sk_metrics.roc_auc_score, average="macro")
roc_auc_weighted = partial(sk_metrics.roc_auc_score, average="weighted")
roc_auc_samples = partial(sk_metrics.roc_auc_score, average="samples")

# roc_auc for multi-class classification,
roc_auc_ovr_micro = partial(
    sk_metrics.roc_auc_score, multi_class="ovr", average="micro"
)
roc_auc_ovr_macro = partial(
    sk_metrics.roc_auc_score, multi_class="ovr", average="macro"
)
roc_auc_ovr_weighted = partial(
    sk_metrics.roc_auc_score, multi_class="ovr", average="weighted"
)
roc_auc_ovr_samples = partial(
    sk_metrics.roc_auc_score, multi_class="ovr", average="samples"
)

roc_auc_ovo_micro = partial(
    sk_metrics.roc_auc_score, multi_class="ovo", average="micro"
)
roc_auc_ovo_macro = partial(
    sk_metrics.roc_auc_score, multi_class="ovo", average="macro"
)
roc_auc_ovo_weighted = partial(
    sk_metrics.roc_auc_score, multi_class="ovo", average="weighted"
)
roc_auc_ovo_samples = partial(
    sk_metrics.roc_auc_score, multi_class="ovo", average="samples"
)
# loss functions
l1_loss = partial(_torch_loss, loss_name="l1_loss")
