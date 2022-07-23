import inspect
import logging
import math
import random
from pprint import pformat
from typing import Any
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from fedsim import scores
from fedsim.utils import apply_on_dict
from fedsim.utils import search_in_submodules

from ...utils.aggregators import AppendixAggregator
from ...utils.aggregators import SerialAggregator


class CentralFLAlgorithm(object):
    def __init__(
        self,
        data_manager,
        metric_logger,
        num_clients,
        sample_scheme,
        sample_rate,
        model_class,
        epochs,
        loss_fn,
        batch_size,
        test_batch_size,
        local_weight_decay,
        slr,
        clr,
        clr_decay,
        clr_decay_type,
        min_clr,
        clr_step_size,
        device,
        log_freq,
        *args,
        **kwargs,
    ):
        grandpa = inspect.getmro(self.__class__)[-2]
        cls = self.__class__

        grandpa_args = set(inspect.signature(grandpa).parameters.keys())
        self_args = set(inspect.signature(cls).parameters.keys())

        added_args = self_args - grandpa_args
        added_args_dict = {key: getattr(self, key) for key in added_args}
        if len(added_args_dict) > 0:
            logging.info("algorithm arguments: " + pformat(added_args_dict))

        self._data_manager = data_manager
        self.num_clients = num_clients
        self.sample_scheme = sample_scheme
        self.sample_count = int(sample_rate * num_clients)
        if not 1 <= self.sample_count <= num_clients:
            raise Exception(
                "invalid client sample size for {}% of {} clients".format(
                    sample_rate, num_clients
                )
            )
        # support functools.partial as model_class
        if hasattr(model_class, "func"):
            model_class_ = getattr(model_class, "func")
        else:
            model_class_ = model_class

        if isinstance(model_class_, str):
            self.model_class = search_in_submodules("fedsim.models", model_class)
        elif issubclass(model_class_, nn.Module):
            self.model_class = model_class
        else:
            raise Exception("incompatiple model!")
        self.epochs = epochs

        if isinstance(loss_fn, str) and hasattr(scores, loss_fn):
            self.loss_fn = getattr(scores, loss_fn)
        else:
            self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.local_weight_decay = local_weight_decay
        self.slr = slr
        self.clr = clr
        self.clr_decay = clr_decay
        self.clr_decay_type = clr_decay_type
        self.min_clr = min_clr
        self.clr_step_size = clr_step_size

        self.metric_logger = metric_logger
        self.device = device
        self.log_freq = log_freq

        self._server_memory: Dict[Hashable, object] = dict()
        self._client_memory: Dict[int, Dict[object]] = {
            k: dict() for k in range(num_clients)
        }

        self.global_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=self.test_batch_size,
                pin_memory=True,
            )
            for key, dataset in self._data_manager.get_global_dataset().items()
        }

        self.oracle_dataset = self._data_manager.get_oracle_dataset()
        self.rounds = 0

        # for internal use only
        self._last_client_sampled: int = None

        self._server_scores = {key: dict() for key in self.global_dataloaders}
        self._client_scores = {
            key: dict() for key in self._data_manager.get_local_splits_names()
        }

    def write_server(self, key, obj):
        self._server_memory[key] = obj

    def write_client(self, client_id, key, obj):
        self._client_memory[client_id][key] = obj

    def read_server(self, key):
        return self._server_memory[key] if key in self._server_memory else None

    def read_client(self, client_id, key):
        if client_id >= self.num_clients:
            raise Exception("invalid client id {} >= {}".format(id, self.num_clients))
        if key in self._client_memory[client_id]:
            return self._client_memory[client_id][key]
        return None

    def _sample_clients(self):
        if self.sample_scheme == "uniform":
            clients = random.sample(range(self.num_clients), self.sample_count)
        elif self.sample_scheme == "sequential":
            last_sampled = (
                -1 if self._last_client_sampled is None else self._last_client_sampled
            )
            clients = [
                (i + 1) % self.num_clients
                for i in range(last_sampled, last_sampled + self.sample_count)
            ]
            self._last_client_sampled = clients[-1]
        else:
            raise NotImplementedError
        return clients

    def _send_to_client(self, client_id):
        return self.send_to_client(client_id=client_id)

    def _send_to_server(self, client_id):
        if self.clr_decay_type == "step":
            decayed_clr = self.clr * (
                self.clr_decay ** (self.rounds // self.clr_step_size)
            )
        elif self.clr_decay_type == "cosine":
            T_i = self.clr_step_size
            T_cur = self.rounds % T_i
            decayed_clr = self.min_clr + 0.5 * (self.clr - self.min_clr) * (
                1 + math.cos(math.pi * T_cur / T_i)
            )

        client_ctx = self.send_to_server(
            client_id,
            self._data_manager.get_local_dataset(client_id),
            self.epochs,
            self.loss_fn,
            self.batch_size,
            decayed_clr,
            self.local_weight_decay,
            self.device,
            ctx=self._send_to_client(client_id),
        )
        if not isinstance(client_ctx, dict):
            raise Exception("client should only return a dict!")
        return {**client_ctx, "client_id": client_id}

    def _receive_from_client(self, client_msg, aggregator):
        client_id = client_msg.pop("client_id")
        return self.receive_from_client(client_id, client_msg, aggregator)

    def _optimize(self, aggregator):
        reports = self.optimize(aggregator)
        # purge aggregated results
        del aggregator
        return reports

    def _report(self, optimize_reports=None, deployment_points=None):
        report_metrics = self.report(
            self.global_dataloaders,
            self.metric_logger,
            self.device,
            optimize_reports,
            deployment_points,
        )
        log_fn = self.metric_logger.add_scalar
        apply_on_dict(report_metrics, log_fn, global_step=self.rounds)
        return report_metrics

    def _train(self, rounds, num_score_report_point=None):
        score_aggregator = AppendixAggregator(max_deque_lenght=num_score_report_point)
        for self.rounds in trange(rounds):
            round_aggregator = SerialAggregator()
            for client_id in self._sample_clients():
                client_msg = self._send_to_server(client_id)
                self._receive_from_client(client_msg, round_aggregator)
            opt_reports = self._optimize(round_aggregator)
            if self.rounds % self.log_freq == 0:
                deploy_poiont = self.deploy()
                score_dict = self._report(opt_reports, deploy_poiont)
                score_aggregator.append_all(score_dict, step=self.rounds)

        # one last report
        if self.rounds % self.log_freq > 0:
            deploy_poiont = self.deploy()
            score_dict = self._report(opt_reports, deploy_poiont)
            score_aggregator.append_all(score_dict, step=self.rounds)
        return score_aggregator.pop_all()

    # API functions

    def train(
        self,
        rounds: int,
        num_score_report_point: Optional[int] = None,
    ) -> Optional[Dict[str, Optional[float]]]:
        r"""loop over the learning pipeline of distributed algorithm for given
        number of rounds.

        Args:
            rounds (int): number of rounds to train.
            num_score_report_point (int): limits num of points to return reports.

        Returns:
            Optional[Dict[str, Union[float]]]: collected score metrics.
        """
        return self._train(rounds=rounds, num_score_report_point=num_score_report_point)

    def get_model_class(self):
        return self.model_class

    def hook_local_score_function(self, split_name, score_name, score_fn):
        self._client_scores[split_name][score_name] = score_fn

    def hook_global_score_function(self, split_name, score_name, score_fn):
        self._server_scores[split_name][score_name] = score_fn

    def get_local_score_functions(self, split_name) -> Dict[str, Any]:
        return self._client_scores[split_name]

    def get_global_score_functions(self, split_name) -> Dict[str, Any]:
        return self._server_scores[split_name]

    # we do not do type hinting, however, the hints for abstract
    # methods are provided to help clarity for users

    # abstract functions

    def send_to_client(self, client_id: int) -> Mapping[Hashable, Any]:
        """returns context to send to the client corresponding to client_id.
            Do not send shared objects like server model if you made any
            before you deepcopy it.

        Args:
            client_id (int): id of the receiving client

        Raises:
            NotImplementedError: abstract class to be implemented by child

        Returns:
            Mapping[Hashable, Any]: the context to be sent in form of a Mapping
        """
        raise NotImplementedError

    def send_to_server(
        self,
        client_id: int,
        datasets: Dict[str, Iterable],
        epochs: int,
        loss_fn: nn.Module,
        batch_size: int,
        lr: float,
        weight_decay: float = 0,
        device: Union[int, str] = "cuda",
        ctx: Optional[Dict[Hashable, Any]] = None,
        *args,
        **kwargs,
    ) -> Mapping[str, Any]:
        """client operation on the recieved information.

        Args:
            client_id (int): id of the client
            datasets (Dict[str, Iterable]): this comes from Data Manager
            epochs (int): number of epochs to train
            loss_fn (nn.Module): either 'ce' (for cross-entropy) or 'mse'
            batch_size (int): training batch_size
            lr (float): client learning rate
            weight_decay (float, optional): weight decay. Defaults to 0.
            device (Union[int, str], optional): Defaults to 'cuda'.
            ctx (Optional[Dict[Hashable, Any]], optional): context reveived.

        Raises:
            NotImplementedError: abstract class to be implemented by child

        Returns:
            Mapping[str, Any]: client context to be sent to the server
        """
        raise NotImplementedError

    def receive_from_client(
        self,
        client_id: int,
        client_msg: Mapping[Hashable, Any],
        aggregator: Any,
    ):
        """receive and aggregate info from selected clients

        Args:
            client_id (int): id of the sender (client)
            client_msg (Mapping[Hashable, Any]): client context that is sent
            aggregator (Any): aggregator instance to collect info

        Raises:
            NotImplementedError: abstract class to be implemented by child
        """
        raise NotImplementedError

    def optimize(self, aggregator: Any) -> Mapping[Hashable, Any]:
        """optimize server mdoel(s) and return metrics to be reported

        Args:
            aggregator (Any): Aggregator instance

        Raises:
            NotImplementedError: abstract class to be implemented by child

        Returns:
            Mapping[Hashable, Any]: context to be reported
        """
        raise NotImplementedError

    def deploy(self) -> Optional[Mapping[Hashable, Any]]:
        """return Mapping of name -> parameters_set to test the model

        Raises:
            NotImplementedError: abstract class to be implemented by child
        """
        raise NotImplementedError

    def report(
        self,
        dataloaders: Dict[str, Any],
        metric_logger: Any,
        device: str,
        optimize_reports: Mapping[Hashable, Any],
        deployment_points: Optional[Mapping[Hashable, torch.Tensor]] = None,
    ) -> Dict[str, Union[int, float]]:
        """test on global data and report info. If a flatten dict of
        str:Union[int,float] is returned from this function the content is
        automatically logged using the metric logger (e.g., tensorboard.SummaryWriter).
        metric_logger is also passed as an input argument for extra
        logging operations (non scalar).

        Args:
            dataloaders (Any): dict of data loaders to test the global model(s)
            metric_logger (Any): the logging object (e.g., SummaryWriter)
            device (str): 'cuda', 'cpu' or gpu number
            optimize_reports (Mapping[Hashable, Any]): dict returned by \
                optimzier
            deployment_points (Mapping[Hashable, torch.Tensor], optional): \
                output of deploy method

        Raises:
            NotImplementedError: abstract class to be implemented by child
        """
        raise NotImplementedError
