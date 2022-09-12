r"""
Dict Ops
--------
"""


def apply_on_dict(dict_obj, fn, return_as_dict=False, *args, **kwargs):
    """Applies an operation defined by ``fn`` on all the entries in a dectionary.

    Args:
        dict_obj (_type_): _description_
        fn (Callable): method to apply on dictionary entries. The signature must be
            ``fn(key, value, *args, **kwargs)``. where ``*args`` and ``**kwargs`` are
            forwarded from ``apply_on_dict`` method to ``fn``.
        return_as_dict (bool, optional): If True a new dictionary with modified entries
            is returned.

    Returns:
        _type_: _description_
    """
    ans = dict()
    if dict_obj is None:
        return
    for key, value in dict_obj.items():
        x = fn(key, value, *args, **kwargs)
        if return_as_dict:
            ans[key] = x
    if return_as_dict:
        return ans
