from typing import Dict

from federation.base_algorithm import BaseAlgorithm

class Algorithm(BaseAlgorithm):
    def __init__(
        self, data_manager: object, num_clients: int, sample_scheme: str, 
        sample_rate: float, slr: float, clr: float, clr_decay: float, 
        clr_decay_type: str, min_clr: float, algorithm_params: Dict, 
        logger: object, verbosity: int
        ) -> None:
        super.__init__(
            data_manager, num_clients, sample_scheme, sample_rate, slr, clr,
            clr_decay, clr_decay_type, min_clr, algorithm_params, logger, 
            verbosity
            )
        
        # TODO: make the model and optimizer here

        # TODO: write_server model and optimizer
    
    def send_to_client(self, client_id: int) -> Dict:
        # TODO: return a copy of the server model
        pass

    def send_to_server(self, client_id: int, dataset: object, ctx: Dict) -> Dict:
        # TODO: creat a data loader

        # TODO: unpack the model from ctx

        # TODO: optimize the model locally

        # TODO: return optimized model parameters and number of training sampoles
        pass

    def receive_from_client(
        self, client_id: int, client_msg: Dict, aggregation_results: Dict
        ):
        # check aggregation_results if not containing aggregated model and samples add
        # purge client info
        pass

    def optimize(self, aggr_results: Dict) -> None:
        # optimize the server model
        pass
    
    def update_deployment_points(self, old_points: Dict) -> Dict:
        # TODO: update all needed for test and report
        pass
    
    def report(self, logger: object):
        # TODO: do inference, use the logger to log
        pass