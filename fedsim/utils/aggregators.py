from collections import deque
from typing import Dict

from fedsim.utils import apply_on_dict


class SerialAggregator(object):
    """Serially aggregats arbitrary number of weighted or unweigted variables."""

    def __init__(self) -> None:
        self._members = dict()

    def _get_pair(self, value, weight):
        if weight is None:
            return value, None
        else:
            return value * weight, weight

    def add(self, key, value, weight=0):
        new_v, new_w = self._get_pair(value, weight)
        sum_v, cur_w = self._members.get(key, (0, 0))
        self._members[key] = (sum_v + new_v, cur_w + new_w)

    def get(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, w = self._members[key]
        if w is None or w == 0:
            return v
        return v / w

    def get_sum(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, _ = self._members[key]
        return v

    def get_weight(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        _, w = self._members[key]
        return w

    def keys(self):
        return self._members.keys()

    def pop(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, w = self._members.pop(key)
        if w is None or w == 0:
            return v
        return v / w

    def items(self):
        for key in self._members.keys():
            yield key, self.get(key)

    def pop_all(self):
        return {key: self.pop(key) for key in list(self._members.keys())}

    def __contains__(self, key):
        return key in self._members


class AppendixAggregator(object):
    def __init__(self, max_deque_lenght=None) -> None:
        self._members = dict()
        self.max_deque_lenght = max_deque_lenght

    def append(self, key, value, weight=1, step=0):
        list_v, list_w, list_s = self._members.get(
            key,
            (
                deque(maxlen=self.max_deque_lenght),  # for values
                deque(maxlen=self.max_deque_lenght),  # for keys
                deque(maxlen=self.max_deque_lenght),  # for steps
            ),
        )
        list_v.append(value)
        list_w.append(weight)
        list_s.append(step)
        if key not in self._members:
            self._members[key] = (list_v, list_w, list_s)

    def append_all(self, entry_dict: Dict[str, float], weight=1, step=0):
        apply_on_dict(entry_dict, self.append, weight=weight, step=step)

    def get(self, key: str, k: int = None):
        r"""fetches the weighted result

        Args:
            key (str): the name of the variable
            k (int, optional): limits the number of points to aggregate.

        Returns:
            Any: the result of the aggregation
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        list_v, list_w, _ = self._members[key]
        if k is None:
            k = len(list_v)
        start_idx = min(k, len(list_v))
        return sum(
            v * w for v, w in zip(list_v[-start_idx:], list_w[-start_idx:])
        ) / sum(list_w[-start_idx:])

    def get_values(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, _, _ = self._members[key]
        return v

    def get_weights(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        _, w, _ = self._members[key]
        return w

    def get_steps(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        _, _, s = self._members[key]
        return s

    def keys(self):
        return self._members.keys()

    def get_list(self, key):
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        return self._members[key]

    def pop(self, key):
        if key not in self._members:
            raise Exception("{} is not in the aggregator".format(key))
        list_v, list_w, _ = self._members.pop(key)
        return sum(v * w for v, w in zip(list_v, list_w)) / sum(list_w)

    def items(self):
        for key in self._members.keys():
            yield key, self.get(key)

    def pop_all(self):
        return {key: self.pop(key) for key in list(self._members.keys())}

    def __contains__(self, key):
        return key in self._members
