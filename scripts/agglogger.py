import os
from typing import List
from typing import Optional
from typing import Any
from logall import Logger
from fedsim.utils.aggregators import AppendixAggregator

class AggLogger(Logger):
    def __init__(self, logdir: str = None, ignore_patterns: List[str] = None, max_len=None) -> None:
        if not logdir:
            import socket
            from datetime import datetime

            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            logdir = os.path.join(
                "runs", current_time + "_" + socket.gethostname()
            )
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.max_len = max_len
        super().__init__(logdir, ignore_patterns)
    
    def register_logger_object(self) -> Any:
        return AppendixAggregator(max_deque_lenght=self.max_len)

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        logger_ob = self.get_logger_object()
        logger_ob.append(key, value, 1, step)

    def get(self, key: str, k: int = None):
        logger_ob = self.get_logger_object()
        return logger_ob.get(key, k)

    def get_dir(self):
        return self._path
