def append_dict_to_dict(new_dict, currecnt_dict=None):
    ans = verify_dict(currecnt_dict)
    for key, item in new_dict.items():
        if key not in ans:
            ans[key] = []
        ans[key].append(item)
    return ans


def add_dict_to_dict(new_dict, currecnt_dict=None, scale=1):
    ans = verify_dict(currecnt_dict)
    for key, item in new_dict.items():
        curr = ans[key] if key in ans else 0
        ans[key] = curr + scale * item
    return ans


def add_in_dict(key, additive, currecnt_dict=None, scale=1):
    ans = verify_dict(currecnt_dict)
    curr = ans[key] if key in ans else 0
    ans[key] = curr + scale * additive
    return ans


def reduce_dict(to_reduct, reduction_fn=lambda x: float(sum(x)) / len(x)):
    return {key: reduction_fn(item) for key, item in to_reduct.items()}


def verify_dict(dict_obj):
    return dict() if dict_obj is None else dict_obj


def apply_on_dict(dict_obj, fn, return_as_dict=False, *args, **kwargs):
    ans = dict()
    if dict_obj is None:
        return
    for key, value in dict_obj.items():
        x = fn(key, value, *args, **kwargs)
        if return_as_dict:
            ans[key] = x
    if return_as_dict:
        return ans
