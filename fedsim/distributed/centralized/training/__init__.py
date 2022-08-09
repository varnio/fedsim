"""
Centralized Training
--------------------

Algorithms for centralized Federated  training.
"""

from .adabest import AdaBest
from .fedavg import FedAvg
from .feddyn import FedDyn
from .fednova import FedNova
from .fedprox import FedProx

__all__ = ["FedAvg", "AdaBest", "FedDyn", "FedNova", "FedProx"]
