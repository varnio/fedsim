"""

FedSim
------

 a simple Federated Learning simulator!

"""

__version__ = "0.1.4"

from . import datasets
from . import distributed
from . import local
from . import models
from . import scores
from . import utils

__all__ = ["datasets", "distributed", "models", "local", "scores", "utils"]
