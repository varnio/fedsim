from typing import List
from typing import Optional
from typing import Any
from logall import Logger
from fedsim.utils.aggregators import AppendixAggregator

class AggLogger(Logger):
    def __init__(self, path: str = None, ignore_patterns: List[str] = None, max_len=None) -> None:
        super().__init__(path, ignore_patterns)
        self.max_len = max_len
    
    def register_logger_object(self) -> Any:
        return AppendixAggregator(max_deque_lenght=self.max_len)

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        logger_ob = self.get_logger_object()
        logger_ob.append(key, value, 1, step)

    def get(self, key: str, k: int = None):
        logger_ob = self.get_logger_object()
        return logger_ob.get(key, k)
