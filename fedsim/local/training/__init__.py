r"""
Local Training and Inference
----------------------------

Provides the basic definitions for local trainign and inference.
"""

from .inference import local_inference
from .step_closures import default_step_closure
from .training import local_train

__all__ = ["local_train", "local_inference", "default_step_closure"]
