r""" A comnbination of sklearn scores and torch.nn loss functions

"""

from sklearn import metrics as sk_metrics
from torch.nn import functional as F


def _get_sk_metric(name: str, **kwargs):
    return lambda input, target, reduction="mean": getattr(sk_metrics, name)(
        y_true=target, y_pred=input, normalize=(reduction == "mean"), **kwargs
    )


accuracy = _get_sk_metric("accuracy_score")
balanced_accuracy = _get_sk_metric("balanced_accuracy_score")
top_k_accuracy = _get_sk_metric("top_k_accuracy_score")
average_precision = _get_sk_metric("average_precision_score")
neg_brier_score = _get_sk_metric("brier_score_loss")
f1_micro = _get_sk_metric("f1_score", average="micro")
f1_macro = _get_sk_metric("f1_score", average="macro")
f1_weighted = _get_sk_metric("f1_score", average="weighted")
f1_samples = _get_sk_metric("f1_score", average="samples")
neg_log_loss = _get_sk_metric("log_loss")

precision_micro = _get_sk_metric("precision_score", average="micro")
precision_macro = _get_sk_metric("precision_score", average="macro")
precision_weighted = _get_sk_metric("precision_score", average="weighted")
precision_samples = _get_sk_metric("precision_score", average="samples")

recall_micro = _get_sk_metric("recall_score", average="micro")
recall_macro = _get_sk_metric("recall_score", average="macro")
recall_weighted = _get_sk_metric("recall_score", average="weighted")
recall_samples = _get_sk_metric("recall_score", average="samples")

jaccard_micro = _get_sk_metric("jaccard_score", average="micro")
jaccard_macro = _get_sk_metric("jaccard_score", average="macro")
jaccard_weighted = _get_sk_metric("jaccard_score", average="weighted")
jaccard_samples = _get_sk_metric("jaccard_score", average="samples")

# roc_auc for binary classification,
roc_auc_micro = _get_sk_metric("roc_auc_score", average="micro")
roc_auc_macro = _get_sk_metric("roc_auc_score", average="macro")
roc_auc_weighted = _get_sk_metric("roc_auc_score", average="weighted")
roc_auc_samples = _get_sk_metric("roc_auc_score", average="samples")

# roc_auc for multi-class classification,
roc_auc_ovr_micro = _get_sk_metric("roc_auc_score", multi_class="ovr", average="micro")
roc_auc_ovr_macro = _get_sk_metric("roc_auc_score", multi_class="ovr", average="macro")
roc_auc_ovr_weighted = _get_sk_metric(
    "roc_auc_score", multi_class="ovr", average="weighted"
)
roc_auc_ovr_samples = _get_sk_metric(
    "roc_auc_score", multi_class="ovr", average="samples"
)

roc_auc_ovo_micro = _get_sk_metric("roc_auc_score", multi_class="ovo", average="micro")
roc_auc_ovo_macro = _get_sk_metric("roc_auc_score", multi_class="ovo", average="macro")
roc_auc_ovo_weighted = _get_sk_metric(
    "roc_auc_score", multi_class="ovo", average="weighted"
)
roc_auc_ovo_samples = _get_sk_metric(
    "roc_auc_score", multi_class="ovo", average="samples"
)
# loss functions
l1_loss = F.l1_loss

cross_entropy = F.cross_entropy
