from tqdm import trange
import random
import yaml
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Union


class BaseAlgorithm(object):

    def __init__(
        self,
        data_manager,
        num_clients,
        sample_scheme,
        sample_rate,
        model,
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
        algorithm_params,
        metric_logger,
        device,
        log_freq,
        verbosity,
    ):

        self._data_manager = data_manager
        self.num_clients = num_clients
        self.sample_scheme = sample_scheme
        self.sample_count = int(sample_rate * num_clients)
        if not 1 <= self.sample_count <= num_clients:
            raise Exception(
                'invalid client sample size for {}% of {} clients'.format(
                    sample_rate, num_clients))
        self.model = model
        self.epochs = epochs
        if loss_fn == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.local_weight_decay = local_weight_decay
        self.slr = slr
        self.clr = clr
        self.clr_decay = clr_decay
        self.clr_decay_type = clr_decay_type
        self.min_clr = min_clr
        self.clr_step_size = clr_step_size

        default_params = self.assign_default_params()
        if default_params is None:
            default_params = dict()
        algorithm_params = {**default_params, **algorithm_params}
        # register algorithm specific parameters
        if algorithm_params is not None:
            for k, v in algorithm_params.items():
                if isinstance(v, str):
                    setattr(self, k, yaml.safe_load(v))
                else:
                    setattr(self, k, v)

        self.metric_logger = metric_logger
        self.device = device
        self.log_freq = log_freq
        self.verbosity = verbosity

        self._server_memory: Dict[Hashable, object] = dict()
        self._client_memory: Dict[int, Dict[object]] = {
            k: dict()
            for k in range(num_clients)
        }

        self.global_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=self.test_batch_size,
                pin_memory=True,
            )
            for key, dataset in
            self._data_manager.get_global_dataset().items()
        }

        self.oracle_dataset = self._data_manager.get_oracle_dataset()
        self.rounds = 0

    def write_server(self, key, obj):
        self._server_memory[key] = obj

    def write_client(self, client_id, key, obj):
        self._client_memory[client_id][key] = obj

    def read_server(self, key):
        return self._server_memory[key] if key in self._server_memory else None

    def read_client(self, client_id, key):
        if client_id >= self.num_clients:
            raise Exception("invalid client id {} >=".format(
                id, self.num_clients))
        if key in self._client_memory[client_id]:
            return self._client_memory[client_id][key]
        return None

    def _sample_clients(self):
        if self.sample_scheme == 'uniform':
            return random.sample(range(self.num_clients), self.sample_count)
        raise NotImplementedError

    def _send_to_client(self, client_id):
        return self.send_to_client(client_id=client_id)

    def _send_to_server(self, client_id):
        if self.clr_decay_type == 'step':
            decayed_clr = self.clr * \
                (self.clr_decay ** (self.rounds // self.clr_step_size))
        elif self.clr_decay_type == 'cosine':
            T_i = self.clr_step_size
            T_cur = self.rounds % T_i
            decayed_clr = self.min_clr + 0.5 * \
                            (self.clr - self.min_clr) * \
                                (1 + math.cos(math.pi * T_cur / T_i))

        client_ctx = self.send_to_server(
            client_id,
            self._data_manager.get_local_dataset(client_id),
            self.epochs,
            self.loss_fn,
            self.batch_size,
            decayed_clr,
            self.local_weight_decay,
            self.device,
            ctx=self._send_to_client(client_id))
        if not isinstance(client_ctx, dict):
            raise Exception('client should only return a dict!')
        return {**client_ctx, 'client_id': client_id}

    def _receive_from_client(self, client_msg, aggregation_results):
        client_id = client_msg.pop('client_id')
        return self.receive_from_client(client_id, client_msg,
                                        aggregation_results)

    def _optimize(self, aggr_results):
        reports = self.optimize(aggr_results)
        # purge aggregated results
        del aggr_results
        return reports

    def _report(self, optimize_reports=None, deployment_points=None):
        self.report(self.global_dataloaders, self.metric_logger, self.device,
                    optimize_reports, deployment_points)

    def _train(self, rounds):
        for self.rounds in trange(rounds):
            aggr_results = dict()
            for client_id in self._sample_clients():
                client_msg = self._send_to_server(client_id)
                self._receive_from_client(client_msg, aggr_results)
            opt_reports = self._optimize(aggr_results)
            if self.rounds % self.log_freq == 0:
                deploy_poiont = self.deploy()
                self._report(opt_reports, deploy_poiont)
        # one last report
        if self.rounds % self.log_freq > 0:
            deploy_poiont = self.deploy()
            self._report(opt_reports, deploy_poiont)

    def train(self, rounds):
        return self._train(rounds=rounds)

    # we do not do type hinting, however, the hints for avstract
    # methods are provided to help clarity for users

    def assign_default_params(self) -> Mapping[Hashable, Any]:
        raise NotImplementedError

    def send_to_client(self, client_id: int) -> Mapping[Hashable, Any]:
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
        device: Union[int, str] = 'cuda',
        ctx: Optional[Dict[Hashable, Any]] = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def receive_from_client(self, client_id: int, client_msg: Mapping[Hashable,
                                                                      Any],
                            aggregation_results: Dict[str, Any]):
        raise NotImplementedError

    def optimize(self, aggr_results: Dict[Hashable,
                                          Any]) -> Mapping[Hashable, Any]:
        raise NotImplementedError

    def deploy(self):
        raise NotImplementedError

    def report(self,
               dataloaders,
               metric_logger: Any,
               device: str,
               optimize_reports: Mapping[Hashable, Any],
               deployment_points: Mapping[Hashable, torch.Tensor] = None):
        raise NotImplementedError
