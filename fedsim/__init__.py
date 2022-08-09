"""

FedSim
------

 a simple Federated Learning simulator!

"""

__version__ = "0.3.1"

from . import datasets
from . import distributed
from . import local
from . import lr_schedulers
from . import models
from . import scores
from . import utils

__all__ = [
    "datasets",
    "distributed",
    "models",
    "local",
    "lr_schedulers",
    "scores",
    "utils",
]
