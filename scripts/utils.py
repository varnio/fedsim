"""
FedSim cli Utils
----------------
"""
import importlib.util
import inspect
import os
from functools import partial
from typing import Dict
from typing import NamedTuple
from typing import OrderedDict
from typing import Tuple

import click
import skopt
import yaml
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real

from fedsim.utils import get_from_module

space_list = [Real, Integer, Categorical]


class ObjectContext(NamedTuple):
    definition: str
    arguments: str
    harguments: skopt.space.Space


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
        raise Exception(f"{definition} is not defined!")
    return definition


# credit goes to
# https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
class OptionEatAll(click.Option):
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def decode_margs(obj_and_args: Tuple):
    obj_and_args = list(obj_and_args)
    # local definition
    if any([len(i) > 1 for i in obj_and_args]):
        obj_name = obj_and_args.pop(0)
        obj_arg_list = obj_and_args
        obj = obj_name
    # included definitions
    else:
        obj = "".join(obj_and_args)
        obj_arg_list = []

    obj_args = dict()
    obj_hargs = OrderedDict()
    while len(obj_arg_list) > 0:
        arg = obj_arg_list.pop(0)
        if ":" in arg:
            slices = arg.split(":")
            key = slices[0]
            if not 2 <= len(slices) <= 3:
                raise Exception(f"{arg} does not seem to be enetered correctly!")
            if len(slices) == 3:
                val = eval(f"{slices[1]}({slices[2].replace('-', ',')})")
                obj_hargs[key] = val
            else:
                val = yaml.safe_load(slices[1])
                obj_args[key] = val
        else:
            raise Exception(f"{arg} is invalid argument!")
    return obj, obj_args, obj_hargs


def ingest_fed_context(
    data_manager,
    algorithm,
    model,
    optimizer,
    local_optimizer,
    lr_scheduler,
    local_lr_scheduler,
    r2r_local_lr_scheduler,
) -> Dict[str, ObjectContext]:
    # decode
    data_manager, data_manager_args, data_manager_hargs = decode_margs(data_manager)
    algorithm, algorithm_args, algorithm_hargs = decode_margs(algorithm)
    model, model_args, model_hargs = decode_margs(model)
    optimizer, optimizer_args, optimizer_hargs = decode_margs(optimizer)
    local_optimizer, local_optimizer_args, local_optimizer_hargs = decode_margs(
        local_optimizer
    )
    lr_scheduler, lr_scheduler_args, lr_scheduler_hargs = decode_margs(lr_scheduler)
    (
        local_lr_scheduler,
        local_lr_scheduler_args,
        local_lr_scheduler_hargs,
    ) = decode_margs(local_lr_scheduler)
    (
        r2r_local_lr_scheduler,
        r2r_local_lr_scheduler_args,
        r2r_local_lr_scheduler_hargs,
    ) = decode_margs(r2r_local_lr_scheduler)
    # find class defs
    data_manager_class = get_definition(
        name=data_manager,
        modules="fedsim.distributed.data_management",
    )
    algorithm_class = get_definition(
        name=algorithm,
        modules=[
            "fedsim.distributed.centralized.training",
            "fedsim.distributed.decentralized.training",
        ],
    )
    model_class = get_definition(
        name=model,
        modules="fedsim.models",
    )
    optimizer_class = get_definition(
        name=optimizer,
        modules="torch.optim",
    )
    local_optimizer_class = get_definition(
        name=local_optimizer,
        modules="torch.optim",
    )
    lr_scheduler_class = get_definition(
        name=lr_scheduler,
        modules="torch.optim.lr_scheduler",
    )
    local_lr_scheduler_class = get_definition(
        name=local_lr_scheduler,
        modules="torch.optim.lr_scheduler",
    )
    r2r_local_lr_scheduler_class = get_definition(
        name=r2r_local_lr_scheduler,
        modules="fedsim.lr_schedulers",
    )
    # raise if algorithm parent signature is overwritten (allow only hparam args)
    grandpa = inspect.getmro(algorithm_class)[-2]
    grandpa_args = set(inspect.signature(grandpa).parameters.keys())
    for alg_arg in algorithm_args:
        if alg_arg in grandpa_args:
            raise Exception(
                f"Not allowed to change parameters of {grandpa} which is "
                f"the parent of algorithm class {algorithm_class}."
                "Check other cli options"
            )

    # partially customize defs
    data_manager_class = partial(data_manager_class, **data_manager_args)
    algorithm_class = partial(algorithm_class, **algorithm_args)
    model_class = partial(model_class, **model_args)
    optimizer_class = partial(optimizer_class, **optimizer_args)
    local_optimizer_class = partial(local_optimizer_class, **local_optimizer_args)
    lr_scheduler_class = partial(lr_scheduler_class, **lr_scheduler_args)
    local_lr_scheduler_class = partial(
        local_lr_scheduler_class,
        **local_lr_scheduler_args,
    )
    r2r_local_lr_scheduler_class = partial(
        r2r_local_lr_scheduler_class,
        **r2r_local_lr_scheduler_args,
    )

    # pack and return as dict
    cfg = OrderedDict(
        data_manager=ObjectContext(
            data_manager_class, data_manager_args, data_manager_hargs
        ),
        algorithm=ObjectContext(algorithm_class, algorithm_args, algorithm_hargs),
        model=ObjectContext(model_class, model_args, model_hargs),
        optimizer=ObjectContext(optimizer_class, optimizer_args, optimizer_hargs),
        local_optimizer=ObjectContext(
            local_optimizer_class, local_optimizer_args, local_optimizer_hargs
        ),
        lr_scheduler=ObjectContext(
            lr_scheduler_class, lr_scheduler_args, lr_scheduler_hargs
        ),
        local_lr_scheduler=ObjectContext(
            local_lr_scheduler_class,
            local_lr_scheduler_args,
            local_lr_scheduler_hargs,
        ),
        r2r_local_lr_scheduler=ObjectContext(
            r2r_local_lr_scheduler_class,
            r2r_local_lr_scheduler_args,
            r2r_local_lr_scheduler_hargs,
        ),
    )
    return cfg


# to use separate log files as suggested at https://stackoverflow.com/a/57774450/9784436
class LogFilter:
    def __init__(self, flow):
        self.flow = flow

    def filter(self, record):
        if record.flow == self.flow:
            return True
