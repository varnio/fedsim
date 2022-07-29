import importlib


def get_from_module(module_name, entry_name):
    module = importlib.import_module(module_name)
    if entry_name in module.__all__:
        return getattr(module, entry_name)
    return None
