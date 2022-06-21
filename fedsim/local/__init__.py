import logging
import os

# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    module[:-3]
    for module in os.listdir(os.path.dirname(__file__))
    if module != "__init__.py" and module[-3:] == ".py"
]
