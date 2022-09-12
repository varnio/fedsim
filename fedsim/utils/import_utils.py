r"""
Import Utils
------------
"""

import importlib


def get_from_module(module_name, entry_name):
    """Imports a module and returns it desired member if existed.

    Args:
        module_name (str): name of the module
        entry_name (str): name of the definition within the module.

    Returns:
        Any: the desired definition in the given module if existed; None otherwise.
    """
    module = importlib.import_module(module_name)
    if hasattr(module, entry_name):
        return getattr(module, entry_name)
    return None
