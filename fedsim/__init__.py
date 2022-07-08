""" a simple Federated Learning simulator!
"""

__version__ = "0.1.2"

import logging
import os

# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# get all classes
__all__ = [
    module[:-3]
    for module in os.listdir(os.path.dirname(__file__))
    if module != "__init__.py" and module[-3:] == ".py"
]

from fedsim import distributed as distributed
from fedsim import local as local
from fedsim import models as models
from fedsim import datasets as datasets


__all__ += ['distributed', 'local', 'models', 'datasets']



