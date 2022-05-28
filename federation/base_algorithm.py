from typing import Dict, Hashable, Optional, Callable
from tqdm import trange
import random
import yaml
import math
from torch import nn
from torch.utils.data import DataLoader


class BaseAlgorithm(object):
    def __init__(
        self, data_manager: object, num_clients: int, sample_scheme: str, 
        sample_rate: float, model: object, epochs: int, loss_fn: str, 
        batch_size: int, test_batch_size: int, local_weight_decay: float, 
        slr: float, clr: float, clr_decay: float, clr_decay_type: str, 
        min_clr: float, clr_step_size: int, algorithm_params: Dict, 
        logger: object, device: str, verbosity: int,
        ) -> None:
        
        self._data_manager = data_manager
        self.num_clients = num_clients
        self.sample_scheme = sample_scheme
        self.sample_count = int(sample_rate * num_clients)
        if not 1 <= self.sample_count <= num_clients:
            raise Exception(
                'invalid client sample size for {}% of {} clients'.format(
                    sample_rate, num_clients
                    )
                )
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
        
        # register algorithm specific parameters
        if algorithm_params is not None:
            for k, v in algorithm_params.items():
                if isinstance(v, str):
                    setattr(self, k, yaml.safe_load(v))
                else:
                    setattr(self, k, v)

        self.logger = logger
        self.device = device
        self.verbosity = verbosity

        self._server_memory: Dict[Hashable, object] = dict()
        self._client_memory: Dict[int, Dict[object]] = {
            k:dict() for k in range(num_clients)
            }
        
        self.global_dataloaders = {
            key:DataLoader(
                dataset, batch_size=self.test_batch_size, pin_memory=True) \
                    for key, dataset in \
                        self._data_manager.get_global_dataset().items()
                    }
    
    def write_server(self, key: Hashable, obj: object) -> None:
        self._server_memory[key] = obj
    
    def write_client(
        self, client_id: int, key: Hashable, obj: object
        ) -> None:
        self._client_memory[client_id][key] = obj
    
    def read_server(self, key: Hashable) -> object:
        if key in self._server_memory:
            return self._server_memory[key]
        return None

    def read_client(self, client_id: int, key: Hashable) -> object:
        if id >= self.num_clients:
            raise Exception(
                "invalid client id {} >=".format(id, self.num_clients)
                )
        if key in self._client_memory[id]:
            return self._client_memory[id][key]
        return None
    
    def _sample_clients(self) -> None:
        if self.sample_scheme == 'uniform':
            return random.sample(range(self.num_clients), self.sample_count)
        raise NotImplementedError
    
    def _send_to_client(self, client_id: int) -> Dict:
        return self.send_to_client(client_id=client_id)
    
    def _send_to_server(self, client_id: int) -> Dict:
        if self.clr_decay_type == 'step':
            decayed_clr = self.clr * (
                self.clr_decay ** (self.rounds // self.clr_step_size)
                )
        elif self.clr_decay_type == 'cosine':
            T_i = self.clr_step_size
            T_cur = self.rounds % T_i
            decayed_clr = self.min_clr + 0.5 * \
                            (self.clr - self.min_clr) * \
                                (1 + math.cos(math.pi * T_cur / T_i))

        client_ctx = self.send_to_server(
            client_id, self._data_manager.get_local_dataset(client_id), 
            self.epochs, self.loss_fn, self.batch_size, decayed_clr, 
            self.local_weight_decay, self.device, 
            ctx=self._send_to_client(client_id))
        if not isinstance(client_ctx, dict):
            raise Exception('client should only return a dict!')
        return {**client_ctx, 'client_id': client_id}

    def _receive_from_client(
        self, client_msg: Dict, aggregation_results: Dict
        ) -> None:
        client_id = client_msg.pop('client_id')
        return self.receive_from_client(
            client_id, client_msg, aggregation_results
            )

    def _optimize(self, aggr_results: Dict) -> None:
        return self.optimize(self.slr, aggr_results)
    
    def _report(self):
        self.report(self.global_dataloaders, self.logger, self.device)

    def _train(self, rounds: int) -> float:
        self._report()
        for self.rounds in trange(rounds):
            aggr_results = dict()
            for client_id in self._sample_clients():
                client_msg = self._send_to_server(client_id)
                self._receive_from_client(client_msg, aggr_results)
            self._optimize(aggr_results)
            self._report()
    
    def train(self, rounds: int) -> float:
        return self._train(rounds=rounds)

    def send_to_client(self, client_id: int) -> Dict:
        raise NotImplementedError

    def send_to_server(
        self, client_id: int, datasets: Dict[str, object], epochs: int, 
        loss_fn: Callable, batch_size: int, lr: float, weight_decay: float = 0, 
        device: str = 'cuda', ctx: Optional[Dict[Hashable, object]] = None,
        ) -> Dict:
        raise NotImplementedError

    def receive_from_client(
        self, client_id: int, client_msg: Dict, aggregation_results: Dict
        ):
        raise NotImplementedError

    def optimize(self, lr: float, aggr_results: Dict) -> None:
        raise NotImplementedError
    
    def report(
        self, dataloaders: Dict[str, object], logger: object, device: str
        ):
        raise NotImplementedError



    