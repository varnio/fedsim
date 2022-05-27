from ctypes import Union
from typing import Dict, Hashable
from tqdm import trange
import random
import yaml


class BaseAlgorithm(object):
    def __init__(
        self, data_manager: object, num_clients: int, sample_scheme: str, 
        sample_rate: float, slr: float, clr: float, clr_decay: float, 
        clr_decay_type: str, min_clr: float, algorithm_params: Dict, 
        logger: object, verbosity: int
        ) -> None:
        
        self._data_manager = data_manager
        self.num_clients = num_clients
        self.sample_scheme = sample_scheme
        self.sample_count = int(sample_rate * num_clients)
        if not 0 <= self.sample_count <= 1:
            raise Exception('invalid client sample size')

        self.slr = slr
        self.clr = clr
        self.clr_decay = clr_decay
        self.clr_decay_type = clr_decay_type
        self.min_clr = min_clr
        
        # register algorithm specific parameters
        if algorithm_params is not None:
            for k, v in algorithm_params.items():
                if isinstance(v, str):
                    setattr(self, k, yaml.safe_load(v))
                else:
                    setattr(self, k, v)

        self.logger = logger
        self.verbosity = verbosity

        self._server_memory: Dict[Hashable, object] = dict()
        self._client_memory: Dict[int, Dict[object]] = {
            k:dict() for k in range(num_clients)
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
        if self.sampling_scheme == 'uniform':
            return random.sample(range(self.num_clients), self.sample_count)
        raise NotImplementedError
    
    def _send_to_client(self, client_id: int, ctx: Dict) -> Dict:
        return self.send_to_client(
            client_id=client_id, 
            data=self._data_manager.get_local_dataset(
                client_id, split='train'
                ), 
            ctx=ctx
            )
    
    def _send_to_server(
        self, client_id: int, dataset: object, ctx: Dict
        ) -> Dict:
        client_ctx = self.send_to_server(client_id, dataset, ctx)
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
    
    def _report(self):
        self.report()

    def _train(self, rounds: int) -> float:
        for _ in trange(rounds):
            aggr_results = dict()
            for client_id in self.sample_clients():
                client_msg = self._send_to_server(
                    client_id, ctx=self._send_to_client(client_id)
                    )
                self._receive_from_client(client_msg, aggr_results)
            self.optimize(aggr_results)
            self._report(self.logger)
    
    def train(self, rounds: int) -> float:
        return self._train(rounds=rounds)

    def send_to_client(self, client_id: int) -> Dict:
        raise NotImplementedError

    def send_to_server(
        self, client_id: int, dataset: object, ctx: Dict
        ) -> Dict:
        raise NotImplementedError

    def receive_from_client(
        self, client_id: int, client_msg: Dict, aggregation_results: Dict
        ):
        raise NotImplementedError

    def optimize(self, aggr_results: Dict) -> None:
        raise NotImplementedError
    
    def update_deployment_points(self, old_points: Dict) -> Dict:
        raise NotImplementedError
    
    def report(self, logger: object):
        raise NotImplementedError



    