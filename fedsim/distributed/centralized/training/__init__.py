import os

from .adabest import AdaBest
from .fedavg import FedAvg
from .fedavgm import FedAvgM
from .feddyn import FedDyn
from .fednova import FedNova
from .fedprox import FedProx

__all__ = [
    module[:-3]
    for module in os.listdir(os.path.dirname(__file__))
    if module != "__init__.py" and module[-3:] == ".py"
]

__all__ += ["FedAvg", "FedAvgM", "AdaBest", "FedDyn", "FedNova", "FedProx"]
