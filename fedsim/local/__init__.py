r"""
Local
-----
"""

from .training import default_step_closure
from .training import local_inference
from .training import local_train

__all__ = ["local_train", "local_inference", "default_step_closure"]
