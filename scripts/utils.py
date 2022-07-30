"""
FedSim cli Utils
----------------
"""
import importlib.util
import inspect
import os
from collections import namedtuple

import yaml

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


def get_context_pool(context, data_manager_class, algorithm_class, model_class):
    dtm_args = dict()
    alg_args = dict()
    mdl_args = dict()
    ClassContext = namedtuple("ClassContext", ["cls", "prefix", "arg_dict"])

    context_pool = dict(
        alg_context=ClassContext(algorithm_class, "a-", alg_args),
        dtm_context=ClassContext(data_manager_class, "d-", dtm_args),
        mdl_context=ClassContext(model_class, "m-", mdl_args),
    )

    def add_arg(key, value, prefix):
        context = list(filter(lambda x: x.prefix == prefix, context_pool.values()))
        if len(context) == 0:
            raise Exception("{} is an invalid argument".format(key))
        else:
            context = context[0]
        if key in inspect.signature(context.cls).parameters.keys():
            context.arg_dict[key] = yaml.safe_load(value)
        else:
            raise Exception(
                "{} is not an argument of {}".format(key, context.cls.__name__)
            )

    i = 0
    while i < len(context.args):
        if context.args[i][:2] != "--":
            raise Exception("unexpected option {}".format(context.args[i]))
        if context.args[i][2] == "-":
            raise Exception(
                "option {} is not valid. No option should starts with ---".format(
                    context.args[i]
                )
            )
        prefix = context.args[i][2:4]
        arg = context.args[i][4:]
        arg = arg.replace("-", "_")
        if i == len(context.args) - 1 or context.args[i + 1][:2] == "--":
            add_arg(arg, "True", prefix)
            i += 1
        else:
            next_arg = context.args[i + 1]
            add_arg(arg, next_arg, prefix)
            i += 2
    return context_pool
