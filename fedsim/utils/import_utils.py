r"""
Import Utils
------------
"""

import importlib


def get_from_module(module_name, entry_name):
    module = importlib.import_module(module_name)
    if hasattr(module, entry_name):
        return getattr(module, entry_name)
    return None
