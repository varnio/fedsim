r"""
Score Utils
-----------
"""


def collect_scores(metric_fn_dict, y_true, y_pred, reduction="mean"):
    answer = {}
    if metric_fn_dict is None:
        return answer
    for name, fn in metric_fn_dict.items():
        answer[name] = fn(y_true, y_pred, reduction=reduction)
    return answer
