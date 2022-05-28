from typing import Callable, Dict, Optional, Hashable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score

import torch

from federation.algorithms.fedavg import Algorithm
from federation.utils import (
    local_train_val, inference, vector_to_parameters_like
    )

class Algorithm(Algorithm):
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
        
        cloud_params = self.read_server('cloud_params')
        self.write_server('avg_params', cloud_params.detach().clone())
        self.write_server('h', torch.zeros_like(cloud_params))
        for client_id in range(num_clients):
            self.write_client(client_id, 'h', torch.zeros_like(cloud_params))
        # oracle read violation, num_clients read violation
        average_sample = len(self.oracle_dataset['train']) / self.num_clients
        self.write_server('average_sample', average_sample)
    
    def assign_default_params(self) -> Optional[Dict[str,object]]:
        return dict(mu=0.01)

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
        params_init = parameters_to_vector(model.parameters()).detach().clone()
        h = self.read_client(client_id, 'h')
        mu_adaptive = self.mu / len(datasets['train']) *\
            self.read_server('average_sample')

        # closure to be performed at each local step
        def step_closure(x, y, model, loss_fn, optimizer, max_grad_norm):
            loss = loss_fn(model(x), y)
            if loss.isnan() or loss.isinf():
                return loss
            # backpropagation
            loss.backward()

            params = parameters_to_vector(model.parameters())
            grad_additive = 0.5 * (params - params_init) - h
            grad_additive_list = vector_to_parameters_like(
                mu_adaptive * grad_additive, model.parameters()
                )

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a

            # Clip gradients
            clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
                )  
            # optimize
            optimizer.step()
            optimizer.zero_grad()
            return loss

        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimize the model locally
        num_train_samples, num_steps, diverged, total_loss = local_train_val(
            model, train_laoder, None, epochs, 0, -1, None, loss_fn, optimizer,
            device, step_closure=step_closure,
            )
        # update local h
        pseudo_grads = (
            params_init - \
            parameters_to_vector(model.parameters()).detach().clone().data
            )
        new_h = h + pseudo_grads
        self.write_client(client_id, 'h', new_h)
        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
        )

    def optimize(self, lr: float, aggr_results: Dict) -> None:
        # get average gradient
        num_sum = aggr_results['num_sum']
        if num_sum > 0:
            param_avg = aggr_results['param_sum'] / num_sum
            
            cloud_params = self.read_server('cloud_params')
            pseudo_grads = cloud_params - param_avg

            counter = aggr_results['counter']
            h = self.read_server('h')
            # read total clients violation
            h = h + counter / self.num_clients *  pseudo_grads
            new_params = param_avg - h

            modified_pseudo_grads = cloud_params - new_params
            # apply sgd
            new_params = cloud_params.data - lr * modified_pseudo_grads.data
            self.write_server('cloud_params', new_params)
            self.write_server('avg_params', param_avg.detach().clone())
            self.write_server('h', h.data)
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
        
        # load cloud stuff
        avg_params = self.read_server('avg_params')
        # copy cloud params to cloud model to send to the client
        vector_to_parameters(
            avg_params.detach().clone().data, cloud_model.parameters()
            )

        # 
        for key, loader in dataloaders.items():
            results = inference(
                cloud_model, loader, 
                {'{}_acc_score'.format(key):accuracy_score}, device=device
                )
            for key, value in results.items():
                logger.add_scalar(key, value, self.rounds)
        