"""
Centralized Training
--------------------

Algorithms for centralized Federated  training.
"""

from .adabest import AdaBest
from .fedavg import FedAvg
from .feddf import FedDF
from .feddyn import FedDyn
from .fednova import FedNova
from .fedprox import FedProx
from .utils import serial_aggregation

__all__ = [
    "FedAvg",
    "AdaBest",
    "FedDyn",
    "FedNova",
    "FedProx",
    "FedDF",
    "serial_aggregation",
]
