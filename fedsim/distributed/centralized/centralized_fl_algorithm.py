r"""
Centralized Federated Learnming Algorithm
-----------------------------------------
"""
import inspect
import random
from functools import partial
from typing import Any
from typing import Callable
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
from fedsim.utils import get_from_module

from ...utils.aggregators import AppendixAggregator
from ...utils.aggregators import SerialAggregator


class CentralFLAlgorithm(object):
    r"""Base class for centralized FL algorithm.

    Args:
            data_manager (``distributed.data_management.DataManager``): data manager
            metric_logger (``logall.Logger``): metric logger for tracking.
            num_clients (int): number of clients
            sample_scheme (``str``): mode of sampling clients. Options are ``'uniform'``
                and ``'sequential'``
            sample_rate (``float``): rate of sampling clients
            model_def (``torch.Module``): definition of for constructing the model
            epochs (``int``): number of local epochs
            criterion_def (``Callable``): loss function defining local objective
            optimizer_def (``Callable``): derfintion of server optimizer
            local_optimizer_def (``Callable``): defintoin of local optimizer
            lr_scheduler_def (``Callable``): definition of lr scheduler of server
                optimizer.
            local_lr_scheduler_def (``Callable``): definition of lr scheduler of local
                optimizer
            r2r_local_lr_scheduler_def (``Callable``): definition to schedule lr that is
                delivered to the clients at each round (deterimined init lr of the
                client optimizer)
            batch_size (int): batch size of the local trianing
            test_batch_size (int): inference time batch size
            device (str): cpu, cuda, or gpu number

    .. note::
        definition of
        * learning rate schedulers, could be any of the ones defined at
        ``fedsim.lr_schedulers``.
        * optimizers, could be any ``torch.optim.Optimizer``.
        * model, could be any ``torch.Module``.
        * criterion, could be any ``fedsim.losses``.

    Architecture:

        .. image:: ../_static/fedlearn.svg
    """

    def __init__(
        self,
        data_manager,
        metric_logger,
        num_clients,
        sample_scheme,
        sample_rate,
        model_def,
        epochs,
        criterion_def,
        optimizer_def,
        local_optimizer_def,
        lr_scheduler_def,
        local_lr_scheduler_def,
        r2r_local_lr_scheduler_def,
        batch_size,
        test_batch_size,
        device,
    ):
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
        # support functools.partial as model_def
        if hasattr(model_def, "func"):
            model_def_ = getattr(model_def, "func")
        else:
            model_def_ = model_def

        if isinstance(model_def_, str):
            self.model_def = get_from_module("fedsim.models", model_def)
        elif issubclass(model_def_, nn.Module):
            self.model_def = model_def
        else:
            raise Exception("incompatiple model!")
        self.epochs = epochs

        if isinstance(criterion_def, str) and hasattr(scores, criterion_def):
            self.criterion_def = getattr(scores, criterion_def)
        else:
            self.criterion_def = criterion_def
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.optimizer_def = optimizer_def
        self.local_optimizer_def = local_optimizer_def
        self.lr_scheduler_def = lr_scheduler_def
        self.local_lr_scheduler_def = local_lr_scheduler_def
        self._train_split_name = "train"

        if r2r_local_lr_scheduler_def is not None:
            # get local lr to build r2r scheduler

            # if partial is used
            if hasattr(local_optimizer_def, "keywords"):
                clr = local_optimizer_def.keywords["lr"]
            # if lr is argumetn
            elif "lr" in inspect.signature(local_optimizer_def).parameters.keys():
                clr = inspect.signature(local_optimizer_def).parameters["lr"].default
            else:
                raise Exception("lr not found in local optimizer class")
            self.r2r_local_lr_scheduler = r2r_local_lr_scheduler_def(init_lr=clr)
        else:
            self.r2r_local_lr_scheduler = None

        self.metric_logger = metric_logger
        self.device = device

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
        if self.r2r_local_lr_scheduler is None:
            local_optimizer_def = self.local_optimizer_def
        else:
            local_optimizer_def = partial(
                self.local_optimizer_def,
                lr=self.r2r_local_lr_scheduler.get_the_last_lr()[0],
            )

        datasets = self._data_manager.get_local_dataset(client_id)
        round_scores = self.get_local_scores()
        client_ctx = self.send_to_server(
            client_id,
            datasets,
            round_scores,
            self.epochs,
            self.criterion_def(),
            self.batch_size,
            self.test_batch_size,
            local_optimizer_def,
            self.local_lr_scheduler_def,
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

    def _report(self, round_scores, optimize_reports=None, deployment_points=None):
        report_metrics = self.report(
            self.global_dataloaders,
            round_scores,
            self.metric_logger,
            self.device,
            optimize_reports,
            deployment_points,
        )
        if self.metric_logger is not None:
            log_fn = self.metric_logger.log_scalar
            apply_on_dict(report_metrics, log_fn, step=self.rounds)
        return report_metrics

    def _train(self, rounds, num_score_report_point=None):
        score_aggregator = AppendixAggregator(max_deque_lenght=num_score_report_point)
        for self.rounds in trange(rounds + 1):
            self._at_round_start()
            round_aggregator = SerialAggregator()
            for client_id in self._sample_clients():
                client_msg = self._send_to_server(client_id)
                self._receive_from_client(client_msg, round_aggregator)
            opt_reports = self._optimize(round_aggregator)
            deploy_poiont = self.deploy()
            round_scores = self.get_global_scores()
            score_dict = self._report(round_scores, opt_reports, deploy_poiont)
            score_aggregator.append_all(score_dict, step=self.rounds)
            self._at_round_end(score_aggregator)

        return score_aggregator.pop_all()

    def _at_round_start(self) -> None:
        self.at_round_start()

    def _at_round_end(self, score_aggregator) -> None:
        if self.r2r_local_lr_scheduler is not None:
            step_args = inspect.signature(self.r2r_local_lr_scheduler.step).parameters
            if "metrics" in step_args:
                trigger_metric = self.r2r_local_lr_scheduler.trigger_metric
                self.r2r_local_lr_scheduler.step(
                    score_aggregator.get(trigger_metric, 1)
                )
            else:
                self.r2r_local_lr_scheduler.step()
        self.at_round_end(score_aggregator)

    def _get_round_scores(self, score_def_deck):
        # filter out the scores that should not be present in the current round
        round_scores = dict()
        for name, definition in score_def_deck.items():
            obj = definition()
            if self.rounds % obj.log_freq == 0:
                round_scores[name] = obj
        return round_scores

    # API functions

    def train(
        self,
        rounds: int,
        num_score_report_point: Optional[int] = None,
        train_split_name="train",
    ) -> Optional[Dict[str, Optional[float]]]:
        r"""loop over the learning pipeline of distributed algorithm for given
        number of rounds.

        .. note::
            * The clients metrics are reported in the form of clients.{metric_name}.
            * The server metrics are reported in the form of
                server.{deployment_point}.{metric_name}

        Args:
            rounds (int): number of rounds to train.
            num_score_report_point (int): limits num of points to return reports.
            train_split_name (str): local split name to perform training on. Defaults
                to 'train'.

        Returns:
            Optional[Dict[str, Union[float]]]: collected score metrics.
        """
        # store default split name
        default_split_name = self._train_split_name
        self._train_split_name = train_split_name
        ans = self._train(
            rounds=rounds,
            num_score_report_point=num_score_report_point,
        )
        # restore default split name
        self._train_split_name = default_split_name
        return ans

    def get_model_def(self):
        return self.model_def

    def get_train_split_name(self):
        return self._train_split_name

    def hook_local_score(self, score_def, score_name, split_name) -> None:
        self._client_scores[split_name][score_name] = score_def

    def hook_global_score(self, score_def, score_name, split_name) -> None:
        self._server_scores[split_name][score_name] = score_def

    def get_local_split_scores(self, split_name) -> Dict[str, Any]:
        return self._get_round_scores(self._client_scores[split_name])

    def get_global_split_scores(self, split_name) -> Dict[str, Any]:
        return self._get_round_scores(self._server_scores[split_name])

    def get_local_scores(self) -> Dict[str, Any]:
        scores = dict()
        for split_name, split in self._client_scores.items():
            split_scores = self._get_round_scores(split)
            if len(split_scores) > 0:
                scores[split_name] = split_scores
        return scores

    def get_global_scores(self) -> Dict[str, Any]:
        scores = dict()
        for split_name, split in self._server_scores.items():
            split_scores = self._get_round_scores(split)
            if len(split_scores) > 0:
                scores[split_name] = split_scores
        return scores

    def at_round_start(self) -> None:
        pass

    def at_round_end(self, score_aggregator: AppendixAggregator) -> None:
        """to inject code at the end of rounds in training loop

        Args:
            score_aggregator (AppendixAggregator): contains the aggregated scores
        """
        pass

    # we do not do type hinting, however, the hints for abstract
    # methods are provided to help clarity for users

    # abstract functions

    def send_to_client(self, client_id: int) -> Mapping[Hashable, Any]:
        """returns context to send to the client corresponding to client_id.

        .. warning::
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
        round_scores: Dict[str, Dict[str, Any]],
        epochs: int,
        criterion: nn.Module,
        train_batch_size: int,
        inference_batch_size: int,
        optimizer_def: Callable,
        lr_scheduler_def: Optional[Callable] = None,
        device: Union[int, str] = "cuda",
        ctx: Optional[Dict[Hashable, Any]] = None,
        *args,
        **kwargs,
    ) -> Mapping[str, Any]:
        """client operation on the recieved information.

        Args:
            client_id (int): id of the client
            datasets (Dict[str, Iterable]): this comes from Data Manager
            round_scores (Dict[str, Dict[str, fedsim.scores.Score]]): dictionary of
                form {'split_name':{'score_name': score_def}} for global scores to
                evaluate at the current round.
            epochs (``int``): number of epochs to train
            criterion (nn.Module): either 'ce' (for cross-entropy) or 'mse'
            train_batch_size (int): training batch_size
            inference_batch_size (int): inference batch_size
            optimizer_def (float): class for constructing the local optimizer
            lr_scheduler_def (float): class for constructing the local lr scheduler
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
        round_scores: Dict[str, Dict[str, Any]],
        metric_logger: Optional[Any],
        device: str,
        optimize_reports: Mapping[Hashable, Any],
        deployment_points: Optional[Mapping[Hashable, torch.Tensor]] = None,
    ) -> Dict[str, Union[int, float]]:
        """test on global data and report info. If a flatten dict of
        str:Union[int,float] is returned from this function the content is
        automatically logged using the metric logger (e.g., logall.TensorboardLogger).
        metric_logger is also passed as an input argument for extra
        logging operations (non scalar).

        Args:
            dataloaders (Any): dict of data loaders to test the global model(s)
            round_scores (Dict[str, Dict[str, fedsim.scores.Score]]): dictionary of
                form {'split_name':{'score_name': score_def}} for global scores to
                evaluate at the current round.
            metric_logger (Any, optional): the logging object
                (e.g., logall.TensorboardLogger)
            device (str): 'cuda', 'cpu' or gpu number
            optimize_reports (Mapping[Hashable, Any]): dict returned by
                optimzier
            deployment_points (Mapping[Hashable, torch.Tensor], optional): \
                output of deploy method

        Raises:
            NotImplementedError: abstract class to be implemented by child
        """
        raise NotImplementedError
