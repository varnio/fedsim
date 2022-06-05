# FedSim
FedSim is a Generic Federated Learning Simulator. It aims to provide the researchers with an easy to develope/maintain simulator for Federated Learning.

# Installation
```bash
pip install fedsim
```


# Usage
```bash
fedsim --help
```


# Supported Datasets
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

# Supported Architectures
- McMahan's 2 layer mlp for MNIST
- McMahan's CNN for CIFAR10 and CIFAR100

# Supported Algorithms
| alias  | paper | 
| ------ | ----- | 
| fedavg | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) |
| fedavg | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) |
| fedavgm| [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) |
| fednova| [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481) |
| fedprox| [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) |
| feddyn | [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) |
| adabest| [Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation](https://arxiv.org/abs/2204.13170) |

# TODO
- [ ] fix the implemntation of adabest (auto avg_m as paper proposed)
- [ ] add implementation of scaffold
- [ ] publish the code
- [ ] add doc (sphinx)


# Contributor's Guide

## Style
- We use `yapf` for formatting the style of the code. Before your merge request:
    - make sure `yapf` is installed.
    - inyour terminal, locate at the root of the project
    - launch the following command: `yapf -ir -vv --no-local-style ./`

- For now, type hinting is only used to avoid confusion at certain points.


## new datasets
New dataset classes should be created under `datasets` directory. Additionally, the datasets should be added to data_manager with appropriate transformations/augmentations.

## new models
New models should be created under `models` directory. If inherited from torch.nn.Module, they would be autmatically recognized by the simlulator and you can call them from the cli.

## new algorithms
New algorithms should be placed under `federation/algorithms/`. The class name should be `Algorithm`. It should inherit `BaseAlgorithm` or its children. A new algorithm class should implement all of the following class methods from `BaseAlgorithm`:
```python
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

    def report(
        self, dataloaders, metric_logger: Any, device: str,
        optimize_reports: Mapping[Hashable, Any], 
        deployment_points: Mapping[Hashable, torch.Tensor] = None
    ):
        raise NotImplementedError
    ```

