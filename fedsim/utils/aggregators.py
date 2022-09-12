"""
Aggregators
-----------
"""
from collections import deque
from typing import Dict

from .dict_ops import apply_on_dict


class SerialAggregator(object):
    """Serially aggregats arbitrary number of weighted or unweigted variables."""

    def __init__(self) -> None:
        self._members = dict()

    def _get_pair(self, value, weight):
        if weight is None:
            return value, None
        else:
            return value * weight, weight

    def add(self, key, value, weight=None):
        """adds a new item to the aggregation

        Args:
            key (Hashable): key of the entry
            value (Any): current value of the entry. Type of this value must support
                addition. Support for division is required if the aggregation is
                weighted.
            weight (float, optional): weight of the current entry. If not specified,
                aggregation becomes unweighted (equal to accumulation). Defaults to
                None.
        """
        new_v, new_w = self._get_pair(value, weight)
        sum_v, cur_w = self._members.get(key, (0, 0))
        if (sum_v is None) or (new_v is None):
            self._members[key] = (sum_v + new_v, None)
        else:
            self._members[key] = (sum_v + new_v, cur_w + new_w)

    def get(self, key):
        """Fetches the current result of the aggregation. If the aggregation is
        weighted the returned value is weighted average of the entry values.

        Args:
            key (Hashable): key to the entry.

        Raises:
            Exception: key does not exist in the aggregator.

        Returns:
            Any: result of the aggregation.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, w = self._members[key]
        if w is None or w == 0:
            return v
        return v / w

    def get_sum(self, key):
        """Fetches the weighted sum (no division).

        Args:
            key (Hashable): key to the entry.

        Raises:
            Exception: key does not exist in the aggregator.

        Returns:
            Any: result of the weighted sum of the entries.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, _ = self._members[key]
        return v

    def get_weight(self, key):
        """Fetches the sum of weights of the weighted averaging.

        Args:
            key (Hashable): key to the entry.

        Raises:
            Exception: key does not exist in the aggregator.

        Returns:
            Any: sum of weights of the aggregation.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        _, w = self._members[key]
        return w

    def keys(self):
        """fetches the keys of entries aggregated so far.

        Returns:
            Iterable: all aggregation keys.
        """
        return self._members.keys()

    def pop(self, key):
        """Similar to ``get`` method except that the entry is removed from the
        aggregator at the end.

        Args:
            key (Hashable): key to the entry.

        Raises:
            Exception: key does not exist in the aggregator.

        Returns:
            Any: result of the aggregation.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, w = self._members.pop(key)
        if w is None or w == 0:
            return v
        return v / w

    def items(self):
        """Generator of (key, result) to get aggregation result of all keys in the
        aggregator.

        Yields:
            Tuple[Hashable, Any]: pair of key, aggregation result.
        """
        for key in self._members.keys():
            yield key, self.get(key)

    def pop_all(self):
        """Collects all the aggregation results in a dictionary and removes everything
        from the aggregator at the end.

        Returns:
            Dict[Hashable, Any]: mapping of key to aggregation result.
        """
        return {key: self.pop(key) for key in list(self._members.keys())}

    def __contains__(self, key):
        return key in self._members


class AppendixAggregator(object):
    """This aggregator hold the results in a deque and performs the aggregation at
    the time querying the results instead. Compared to SerialAggregator provides the
    flexibility of aggregating within a certain number of past entries.

    Args:
        max_deque_lenght (int, optional): maximum lenght of deque to hold the
            aggregation entries. Defaults to None.

    """

    def __init__(self, max_deque_lenght=None) -> None:
        self._members = dict()
        self.max_deque_lenght = max_deque_lenght

    def append(self, key, value, weight=1, step=0):
        """Appends a new weighted entry timestamped by step.

        Args:
            key (Hashable): key to the aggregation entry.
            value (Any): value of the aggregation entry.
            weight (int, optional): weight of the aggregation for the current entry.
                Defaults to 1.
            step (int, optional): timestamp of the current entry. Defaults to 0.
        """
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
        """To apply ``append`` on several entries given by a dictionary.

        Args:
            entry_dict (Dict[Hashable, Any]): dictionary of the entries.
            weight (int, optional): weight of the entries. Defaults to 1.
            step (int, optional): timestamp of the current entries. Defaults to 0.
        """
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
        list_v = list(list_v)
        list_w = list(list_w)
        if k is None:
            k = len(list_v)
        start_idx = min(k, len(list_v))
        return sum(
            v * w for v, w in zip(list_v[-start_idx:], list_w[-start_idx:])
        ) / sum(list_w[-start_idx:])

    def get_values(self, key):
        """fetches the values of the aggregation.

        Args:
            key (Hashable): aggregation key.

        Raises:
            Exception: key not in the aggregator.

        Returns:
            List[Any]: list of values appended up to the maximum lenght of the
            internal deque.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        v, _, _ = self._members[key]
        return v

    def get_weights(self, key):
        """fetches the weights of the aggregation.

        Args:
            key (Hashable): aggregation key.

        Raises:
            Exception: key not in the aggregator.

        Returns:
            List[Any]: list of weights appended up to the maximum lenght of the
            internal deque.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        _, w, _ = self._members[key]
        return w

    def get_steps(self, key):
        """fetches the timestamps of the aggregation.

        Args:
            key (Hashable): aggregation key.

        Raises:
            Exception: key not in the aggregator.

        Returns:
            List[Any]: list of timestamps appended up to the maximum lenght of the
            internal deque.
        """
        if key not in self._members:
            raise Exception(f"{key} is not in the aggregator")
        _, _, s = self._members[key]
        return s

    def keys(self):
        """fetches the keys of entries aggregated so far.

        Returns:
            Iterable: all aggregation keys.
        """
        return self._members.keys()

    def pop(self, key):
        """Similar to ``get`` method except that the entry is removed from the
        aggregator at the end.

        Args:
            key (Hashable): key to the entry.

        Raises:
            Exception: key does not exist in the aggregator.

        Returns:
            Any: result of the aggregation.
        """
        if key not in self._members:
            raise Exception("{} is not in the aggregator".format(key))
        list_v, list_w, _ = self._members.pop(key)
        return sum(v * w for v, w in zip(list_v, list_w)) / sum(list_w)

    def items(self):
        """Generator of (key, result) to get aggregation result of all keys in the
        aggregator.

        Yields:
            Tuple[Hashable, Any]: pair of key, aggregation result.
        """
        for key in self._members.keys():
            yield key, self.get(key)

    def pop_all(self):
        """Collects all the aggregation results in a dictionary and removes everything
        from the aggregator at the end.

        Returns:
            Dict[Hashable, Any]: mapping of key to aggregation result.
        """
        return {key: self.pop(key) for key in list(self._members.keys())}

    def __contains__(self, key):
        return key in self._members
