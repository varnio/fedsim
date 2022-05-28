from copy import deepcopy
from typing import Callable, Dict, Optional, Hashable
from matplotlib.pyplot import step
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import accuracy_score

from federation.base_algorithm import BaseAlgorithm
import utils
from federation.utils import (
    local_train_val, inference, vector_to_parameters_like
    )

class Algorithm(BaseAlgorithm):
    def __init__(
        self, data_manager: object, num_clients: int, sample_scheme: str, 
        sample_rate: float, model: object, epochs: int, loss_fn: str, 
        batch_size: int, test_batch_size: int, local_weight_decay: float, 
        slr: float, clr: float, clr_decay: float, clr_decay_type: str, 
        min_clr: float, clr_step_size: int, algorithm_params: Dict, 
        logger: object, device: str, verbosity: int,
        ) -> None:
        super(Algorithm, self).__init__(
            data_manager, num_clients, sample_scheme, sample_rate, model, 
            epochs, loss_fn, batch_size, test_batch_size, local_weight_decay, 
            slr, clr, clr_decay, clr_decay_type, min_clr, clr_step_size,
            algorithm_params, logger, device, verbosity
            )
        
        model_class = utils.get_from_module(
            'models', 'mcmahan_nets', model
        )
        
        # make mode and optimizer
        model = model_class().to(self.device)
        params = deepcopy(
            parameters_to_vector(model.parameters()).clone().detach()
            )
        optimizer = SGD(params=model.parameters(), lr=slr)
        # write model and optimizer to server
        self.write_server('cloud_model', model)
        self.write_server('cloud_params', params)
        self.write_server('optimizer', optimizer)
    
    def assign_default_params(self) -> Optional[Dict[str,object]]:
        return None
    
    def send_to_client(self, client_id: int) -> Dict:
        # since fedavg broadcast the same model to all selected clients, 
        # the argument client_id is not used

        # load cloud stuff
        cloud_params = self.read_server('cloud_params')
        cloud_model = self.read_server('cloud_model')
        
        # copy cloud params to cloud model to send to the client
        vector_to_parameters(
            cloud_params.detach().clone().data, cloud_model.parameters()
            )
        # return a copy of the cloud model
        return dict(cloud_model=cloud_model)

    def send_to_server(
        self, client_id: int, datasets: Dict[str, object], epochs: int, 
        loss_fn: Callable, batch_size: int, lr: float, weight_decay: float = 0, 
        device: str = 'cuda', ctx: Optional[Dict[Hashable, object]] = None,
        ) -> Dict:
        # create train data loader
        train_laoder = DataLoader(
            datasets['train'], batch_size=batch_size, shuffle=False,
            )
        model = ctx['cloud_model']
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimize the model locally
        num_train_samples, num_steps, diverged, total_loss = local_train_val(
            model, train_laoder, None, epochs, 0, -1, None, loss_fn, optimizer,
            device
            )
        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
        )

    def receive_from_client(
        self, client_id: int, client_msg: Dict, aggregation_results: Dict
        ) -> None:
        num_samples = client_msg['num_samples']
        if len(aggregation_results) == 0:
            aggregation_results['param_sum'] = deepcopy(
                client_msg['local_params'].clone().detach()
                ) * num_samples
            aggregation_results['num_sum'] = num_samples
            aggregation_results['counter'] = 1
        else:
            aggregation_results['param_sum'].data += num_samples * \
                client_msg['local_params'].detach().clone().data
            aggregation_results['num_sum'] += num_samples
            aggregation_results['counter'] += 1 
        # purge client info
        del client_msg

    def optimize(self, lr: float, aggr_results: Dict) -> None:
        # get average gradient
        num_sum = aggr_results['num_sum']
        if num_sum > 0:
            param_avg = aggr_results['param_sum'] / num_sum
            
            cloud_params = self.read_server('cloud_params')
            pseudo_grads = cloud_params - param_avg
            # apply sgd
            new_params = cloud_params.data - lr * pseudo_grads.data
            self.write_server('cloud_params', new_params)
            # purge aggregated results
            del param_avg
            del aggr_results
    
    def report(
        self, dataloaders: Dict[str, object], logger: object, device: str
        ):
        # load cloud stuff
        cloud_params = self.read_server('cloud_params')
        cloud_model = self.read_server('cloud_model')
        # copy cloud params to cloud model to send to the client
        vector_to_parameters(
            cloud_params.detach().clone().data, cloud_model.parameters()
            )

        # 
        for key, loader in dataloaders.items():
            results = inference(
                cloud_model, loader, 
                {'{}_acc_score'.format(key):accuracy_score}, device=device
                )
            for key, value in results.items():
                logger.add_scalar(key, value, self.rounds)