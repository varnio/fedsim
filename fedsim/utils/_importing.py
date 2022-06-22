import importlib
import inspect


def get_from_module(module_name, submodule_name, class_name):
    module = importlib.import_module(module_name)
    if submodule_name in module.__all__:
        sub_module = importlib.import_module(
            "{}.{}".format(module_name, submodule_name)
        )
        return getattr(sub_module, class_name)
    raise NotImplementedError


def search_in_submodules(module_name, object_name):
    module = importlib.import_module(module_name)
    for submodule_name in module.__all__:
        sub_module = importlib.import_module(
            "{}.{}".format(module_name, submodule_name)
        )
        for name, obj in inspect.getmembers(
            sub_module,
            lambda x: inspect.isclass(x) or inspect.isfunction(x),
        ):
            if name == object_name:
                return obj
    return None
