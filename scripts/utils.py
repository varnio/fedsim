"""
FedSim cli Utils
----------------
"""
import importlib.util
import os

from fedsim.utils import get_from_module


def parse_class_from_file(s: str) -> object:
    f, c = s.split(":")
    path = f + ".py"
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location(
            os.path.dirname(path).replace(os.sep, "."),
            path,
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, c)

    return None


def get_definition(name, modules):
    if not isinstance(modules, list):
        modules = [
            modules,
        ]

    if ":" in name:
        definition = parse_class_from_file(name)
    else:
        for module in modules:
            definition = get_from_module(
                module,
                name,
            )
            if definition is not None:
                break
    if definition is None:
        raise Exception(f"{definition} is not a defined!")
    return definition
