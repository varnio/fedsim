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
from fedsim.utils import AppendixAggregator
from fedsim.utils import SerialAggregator
from fedsim.utils import Storage
from fedsim.utils import apply_on_dict
from fedsim.utils import get_from_module


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
        ``torch.optim.lr_scheduler`` or any other that implements step and get_last_lr
        methods.
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
        sample_count = int(sample_rate * num_clients)
        if not 1 <= sample_count <= num_clients:
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
            model_def = get_from_module("fedsim.models", model_def)
        elif issubclass(model_def_, nn.Module):
            model_def = model_def
        else:
            raise Exception("incompatiple model!")

        if isinstance(criterion_def, str) and hasattr(scores, criterion_def):
            criterion_def = getattr(scores, criterion_def)
        else:
            criterion_def = criterion_def

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
            # make a dummy optimizer so to directly use pytorch lr schedulers
            dummy_params = [
                torch.tensor([1.0, 1.0], requires_grad=True),
            ]
            dummy_optimizer = torch.optim.SGD(params=dummy_params, lr=clr)
            r2r_scheduler = r2r_local_lr_scheduler_def(dummy_optimizer)

            def last_private_lr(sch):
                return sch._last_lr

            if not hasattr(r2r_scheduler, "get_last_lr"):
                r2r_scheduler.get_last_lr = partial(last_private_lr, r2r_scheduler)

            r2r_local_lr_scheduler = r2r_scheduler
            dummy_optimizer.step()
        else:
            r2r_local_lr_scheduler = None

        global_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=test_batch_size,
                pin_memory=True,
            )
            for key, dataset in data_manager.get_global_dataset().items()
        }

        oracle_dataset = data_manager.get_oracle_dataset()

        self._psm = Storage()  # private server memory
        self._psm.write("data_manager", data_manager)
        self._psm.write("num_clients", num_clients)
        self._psm.write("sample_count", sample_count)
        self._psm.write("sample_scheme", sample_scheme)
        self._psm.write("test_batch_size", test_batch_size)
        self._psm.write("optimizer_def", optimizer_def)
        self._psm.write("lr_scheduler_def", lr_scheduler_def)
        self._psm.write("r2r_local_lr_scheduler", r2r_local_lr_scheduler)
        self._psm.write("criterion_def", criterion_def)
        self._psm.write("model_def", model_def)
        self._psm.write("metric_logger", metric_logger)
        self._psm.write("device", device)
        self._psm.write("global_dataloaders", global_dataloaders)
        self._psm.write("oracle_dataset", oracle_dataset)
        self._psm.write("rounds", 0)
        self._psm.write("last_client_sampled", None)

        self._pcm = Storage()  # private client memory
        self._pcm.write("epochs", epochs)
        self._pcm.write("batch_size", batch_size)
        self._pcm.write("test_batch_size", test_batch_size)
        self._pcm.write("criterion_def", criterion_def)
        self._pcm.write("local_optimizer_def", local_optimizer_def)
        self._pcm.write("local_lr_scheduler_def", local_lr_scheduler_def)
        self._pcm.write("device", device)

        # this is over written in train method
        self._train_split_name = "train"

        self._server_memory = Storage()
        self._client_memory = {k: Storage() for k in range(num_clients)}
        self._server_scores = {key: dict() for key in global_dataloaders}
        self._client_scores = {
            key: dict() for key in data_manager.get_local_splits_names()
        }

    def _sample_clients(self):
        sample_scheme = self._psm.read("sample_scheme")
        sample_count = self._psm.read("sample_count")
        num_clients = self._psm.read("num_clients")
        last_client_sampled = self._psm.read("last_client_sampled")

        if sample_scheme == "uniform":
            clients = random.sample(range(num_clients), sample_count)
        elif sample_scheme == "sequential":
            last_sampled = -1 if last_client_sampled is None else last_client_sampled
            clients = [
                (i + 1) % num_clients
                for i in range(last_sampled, last_sampled + sample_count)
            ]
            self._psm.write("last_client_sampled", clients[-1])
        else:
            raise NotImplementedError
        return clients

    def _send_to_client(self, client_id):
        return self.send_to_client(self._server_memory, client_id=client_id)

    def _send_to_server(self, client_id):
        r2r_local_lr_scheduler = self._psm.read("r2r_local_lr_scheduler")
        data_manager = self._psm.read("data_manager")
        rounds = self._psm.read("rounds")

        epochs = self._pcm.read("epochs")
        batch_size = self._pcm.read("batch_size")
        test_batch_size = self._pcm.read("test_batch_size")
        local_optimizer_def = self._pcm.read("local_optimizer_def")
        criterion_def = self._pcm.read("criterion_def")
        local_lr_scheduler_def = self._pcm.read("local_lr_scheduler_def")
        device = self._pcm.read("device")

        if r2r_local_lr_scheduler is None:
            local_optimizer_def = local_optimizer_def
        else:
            local_optimizer_def = partial(
                local_optimizer_def,
                lr=r2r_local_lr_scheduler.get_last_lr()[0],
            )

        datasets = data_manager.get_local_dataset(client_id)
        round_scores = self.get_local_scores()
        storage = self._client_memory[client_id]
        train_split_name = self.get_train_split_name()

        client_ctx = self.send_to_server(
            client_id,
            rounds,
            storage,
            datasets,
            train_split_name,
            round_scores,
            epochs,
            criterion_def(),
            batch_size,
            test_batch_size,
            local_optimizer_def,
            local_lr_scheduler_def,
            device,
            ctx=self._send_to_client(client_id),
        )
        if not isinstance(client_ctx, dict):
            raise Exception("client should only return a dict!")
        return {**client_ctx, "client_id": client_id}

    def _receive_from_client(self, client_msg, aggregator):
        client_id = client_msg.pop("client_id")
        train_split_name = self.get_train_split_name()
        return self.receive_from_client(
            self._server_memory, client_id, client_msg, train_split_name, aggregator
        )

    def _optimize(self, aggregator):
        reports = self.optimize(self._server_memory, aggregator)
        # purge aggregated results
        del aggregator
        return reports

    def _report(self, round_scores, optimize_reports=None, deployment_points=None):
        global_dataloaders = self._psm.read("global_dataloaders")
        metric_logger = self._psm.read("metric_logger")
        rounds = self._psm.read("rounds")
        device = self._psm.read("device")

        report_metrics = self.report(
            self._server_memory,
            global_dataloaders,
            rounds,
            round_scores,
            metric_logger,
            device,
            optimize_reports,
            deployment_points,
        )
        if metric_logger is not None:
            log_fn = metric_logger.log_scalar
            apply_on_dict(report_metrics, log_fn, step=rounds)
        return report_metrics

    def _train(self, rounds, num_score_report_point=None):
        diverged = False
        cur_round = self._psm.read("rounds")
        score_aggregator = AppendixAggregator(max_deque_lenght=num_score_report_point)
        for round_num in trange(rounds + 1):
            self._at_round_start()
            round_aggregator = SerialAggregator()
            for client_id in self._sample_clients():
                client_msg = self._send_to_server(client_id)
                success = self._receive_from_client(client_msg, round_aggregator)
                # signal divergence
                if not success:
                    diverged = True
                    break
            # check for divergence, early return
            if diverged:
                return score_aggregator.pop_all()
            # optimzie
            opt_reports = self._optimize(round_aggregator)
            deploy_poiont = self.deploy(self._server_memory)
            round_scores = self.get_global_scores()
            score_dict = self._report(round_scores, opt_reports, deploy_poiont)
            score_aggregator.append_all(score_dict, step=cur_round)
            self._at_round_end(score_aggregator)
            self._psm.write("rounds", cur_round + round_num + 1)

        return score_aggregator.pop_all()

    def _at_round_start(self) -> None:
        self.at_round_start(self._server_memory)

    def _at_round_end(self, score_aggregator) -> None:
        r2r_local_lr_scheduler = self._psm.read("r2r_local_lr_scheduler")
        if r2r_local_lr_scheduler is not None:
            step_args = inspect.signature(r2r_local_lr_scheduler.step).parameters
            # TODO: metrics in step_args is deprecated, remove in future releases
            if "metrics" in step_args:
                trigger_metric = r2r_local_lr_scheduler.trigger_metric
                r2r_local_lr_scheduler.step(score_aggregator.get(trigger_metric, 1))
            else:
                r2r_local_lr_scheduler.step()
        self.at_round_end(self._server_memory, score_aggregator)

    def _get_round_scores(self, score_def_deck):
        # filter out the scores that should not be present in the current round
        rounds = self._psm.read("rounds")
        round_scores = dict()
        for name, definition in score_def_deck.items():
            obj = definition()
            if rounds % obj.log_freq == 0:
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
        model_def = self._psm.read("model_def")
        return model_def

    def get_server_storage(self):
        return self._server_memory

    def get_server_private_storage(self, verbose=True):
        if verbose:
            print(
                "Warning: private server's storage is fetched! "
                "This is most probably due to a violation!"
            )
        return self._psm

    def get_round_number(self):
        return self._psm.read("rounds")

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

    # we do not do type hinting, however, the hints for abstract
    # methods are provided to help clarity for users

    # optional methods
    def at_round_start(self, server_storage: Storage) -> None:
        """to inject code at the beginning of rounds in training loop
        Args:
            server_storage (Storage): server storage object.
        """
        pass

    def at_round_end(
        self,
        server_storage: Storage,
        score_aggregator: AppendixAggregator,
    ) -> None:
        """to inject code at the end of rounds in training loop

        Args:
            server_storage (Storage): server storage object.
            score_aggregator (AppendixAggregator): contains the aggregated scores
        """
        pass

    # abstract methods
    def send_to_client(self, server_storage, client_id: int) -> Mapping[Hashable, Any]:
        """returns context to send to the client corresponding to client_id.

        .. warning::
            Do not send shared objects like server model if you made any
            before you deepcopy it.

        Args:
            server_storage (Storage): server storage object.
            client_id (int): id of the receiving client

        Raises:
            NotImplementedError: abstract class to be implemented by child

        Returns:
            Mapping[Hashable, Any]: the context to be sent in form of a Mapping
        """
        raise NotImplementedError(
            f"Algorithm [{type(self).__name__}] is missing the"
            f"required 'send_to_client' function"
        )

    def send_to_server(
        self,
        id: int,
        rounds: int,
        storage: Dict[Hashable, Any],
        datasets: Dict[str, Iterable],
        train_split_name: str,
        metrics: Dict[str, Dict[str, Any]],
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
            id (int): id of the client
            rounds (int): global round number
            storage (Storage): storage object of the client
            datasets (Dict[str, Iterable]): this comes from Data Manager
            train_split_name (str): string containing name of the training split
            metrics: Dict[str, Dict[str, Any]]: dictionary of
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

        Returns:
            Mapping[str, Any]: client context to be sent to the server

        """
        raise NotImplementedError(
            "Algorithm is missing the required 'send_to_client' function"
        )

    def receive_from_client(
        self,
        server_storage: Storage,
        client_id: int,
        client_msg: Mapping[Hashable, Any],
        train_split_name: str,
        aggregator,
    ) -> bool:
        """receive and aggregate info from selected clients

        Args:
            server_storage (Storage): server storage object.
            client_id (int): id of the sender (client)
            client_msg (Mapping[Hashable, Any]): client context that is sent.
            train_split_name (str): name of the training split on clients.
            aggregator (SerialAggregator): aggregator instance to collect info.

        Returns:
            bool: success of the aggregation.

        Raises:
            NotImplementedError: abstract class to be implemented by child
        """
        raise NotImplementedError

    def optimize(
        self, server_storage: Storage, aggregator: SerialAggregator
    ) -> Mapping[Hashable, Any]:
        """optimize server mdoel(s) and return metrics to be reported

        Args:
            server_storage (Storage): server storage object.
            aggregator (SerialAggregator): Aggregator instance

        Raises:
            NotImplementedError: abstract class to be implemented by child

        Returns:
            Mapping[Hashable, Any]: context to be reported
        """
        raise NotImplementedError

    def deploy(self, server_storage: Storage) -> Optional[Mapping[Hashable, Any]]:
        """return Mapping of name -> parameters_set to test the model

        Args:
            server_storage (Storage): server storage object.
        """
        raise NotImplementedError

    def report(
        self,
        server_storage: Storage,
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
            server_storage (Storage): server storage object.
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
