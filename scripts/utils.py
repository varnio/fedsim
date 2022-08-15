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
    if name is None:
        return None
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
        raise Exception(f"{name} is not defined in {modules}!")
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
    if obj_and_args is None:
        return None, None, None
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
    criterion,
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
    criterion, criterion_args, criterion_hargs = decode_margs(criterion)
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
    data_manager_def = get_definition(
        name=data_manager,
        modules="fedsim.distributed.data_management",
    )
    algorithm_def = get_definition(
        name=algorithm,
        modules=[
            "fedsim.distributed.centralized.training",
            "fedsim.distributed.decentralized.training",
        ],
    )
    model_def = get_definition(
        name=model,
        modules="fedsim.models",
    )
    criterion_def = get_definition(
        name=criterion,
        modules="fedsim.losses",
    )
    optimizer_def = get_definition(
        name=optimizer,
        modules="torch.optim",
    )
    local_optimizer_def = get_definition(
        name=local_optimizer,
        modules="torch.optim",
    )
    lr_scheduler_def = get_definition(
        name=lr_scheduler,
        modules="fedsim.lr_schedulers",
    )
    local_lr_scheduler_def = get_definition(
        name=local_lr_scheduler,
        modules="fedsim.lr_schedulers",
    )
    r2r_local_lr_scheduler_def = get_definition(
        name=r2r_local_lr_scheduler,
        modules="fedsim.lr_schedulers",
    )
    # raise if algorithm parent signature is overwritten (allow only hparam args)
    grandpa = inspect.getmro(algorithm_def)[-2]
    grandpa_args = set(inspect.signature(grandpa).parameters.keys())
    for alg_arg in algorithm_args:
        if alg_arg in grandpa_args:
            raise Exception(
                f"Not allowed to change parameters of {grandpa} which is "
                f"the parent of algorithm class {algorithm_def}."
                "Check other cli options"
            )

    # partially customize defs
    if data_manager_def is not None:
        data_manager_def = partial(data_manager_def, **data_manager_args)
    if algorithm_def is not None:
        algorithm_def = partial(algorithm_def, **algorithm_args)
    if model_def is not None:
        model_def = partial(model_def, **model_args)
    if criterion_def is not None:
        criterion_def = partial(criterion_def, **criterion_args)
    if optimizer_def is not None:
        optimizer_def = partial(optimizer_def, **optimizer_args)
    if local_optimizer_def is not None:
        local_optimizer_def = partial(local_optimizer_def, **local_optimizer_args)
    if lr_scheduler_def is not None:
        lr_scheduler_def = partial(lr_scheduler_def, **lr_scheduler_args)
    if local_lr_scheduler_def is not None:
        local_lr_scheduler_def = partial(
            local_lr_scheduler_def,
            **local_lr_scheduler_args,
        )
    if r2r_local_lr_scheduler_def is not None:
        r2r_local_lr_scheduler_def = partial(
            r2r_local_lr_scheduler_def,
            **r2r_local_lr_scheduler_args,
        )

    # pack and return as dict
    cfg = OrderedDict(
        data_manager=ObjectContext(
            data_manager_def, data_manager_args, data_manager_hargs
        ),
        algorithm=ObjectContext(algorithm_def, algorithm_args, algorithm_hargs),
        model=ObjectContext(model_def, model_args, model_hargs),
        criterion=ObjectContext(criterion_def, criterion_args, criterion_hargs),
        optimizer=ObjectContext(optimizer_def, optimizer_args, optimizer_hargs),
        local_optimizer=ObjectContext(
            local_optimizer_def, local_optimizer_args, local_optimizer_hargs
        ),
        lr_scheduler=ObjectContext(
            lr_scheduler_def, lr_scheduler_args, lr_scheduler_hargs
        ),
        local_lr_scheduler=ObjectContext(
            local_lr_scheduler_def,
            local_lr_scheduler_args,
            local_lr_scheduler_hargs,
        ),
        r2r_local_lr_scheduler=ObjectContext(
            r2r_local_lr_scheduler_def,
            r2r_local_lr_scheduler_args,
            r2r_local_lr_scheduler_hargs,
        ),
    )
    return cfg


def ingest_scores(score_tuple):
    score_tuple = set(score_tuple)  # to get rid of identical scores
    score_objs = []
    for score in score_tuple:
        score_name, score_args, score_hargs = decode_margs(score)
        score_def = get_definition(
            name=score_name,
            modules="fedsim.scores",
        )
        score_objs.append(
            ObjectContext(partial(score_def, **score_args), score_args, score_hargs)
        )
    return score_objs


def validate_score(score, possible_split_names, mode="local"):
    if "split" in score.keywords:
        split_name = score.keywords["split"]
        if split_name not in possible_split_names:
            raise Exception(
                f"{split_name} is not provided by data manager as a "
                f"{mode} data split (possible choices are {possible_split_names})."
            )
    elif "split" in inspect.signature(score.func).parameters.keys():
        default_split = inspect.signature(score.func).parameters["split"].default
        if default_split not in possible_split_names:
            raise Exception(
                f"default split of local score is not provide by data manager as a "
                f"{mode} data split (possible choices are {possible_split_names})."
            )
        split_name = default_split
    else:
        raise Exception("can't find split name")

    if "score_name" in score.keywords:
        score_name = score.keywords["score_name"]
    elif "score_name" in inspect.signature(score.func).parameters.keys():
        default_score = inspect.signature(score.func).parameters["score_name"].default
        score_name = default_score

    return split_name, score_name


# to use separate log files as suggested at https://stackoverflow.com/a/57774450/9784436
class LogFilter:
    def __init__(self, flow):
        self.flow = flow

    def filter(self, record):
        if record.flow == self.flow:
            return True
