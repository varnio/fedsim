from inspect import signature


def get_metric_scores(metric_fn_dict, y_true, y_pred):
    answer = {}
    if metric_fn_dict is None:
        return answer
    for name, fn in metric_fn_dict.items():
        args = dict()
        if "y_true" in signature(fn).parameters:
            args["y_true"] = y_true
        elif "target" in signature(fn).parameters:
            args["target"] = y_true
        else:
            raise NotImplementedError
        if "y_pred" in signature(fn).parameters:
            args["y_pred"] = y_pred
        elif "input" in signature(fn).parameters:
            args["input"] = y_pred
        else:
            raise NotImplementedError
        answer[name] = fn(**args)
    return answer
