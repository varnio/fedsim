class SerialAggregator(object):
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
            raise Exception("{} is not in the aggregator".format(key))
        v, w = self._members[key]
        if w is None or w == 0:
            return v
        return v / w

    def get_sum(self, key):
        if key not in self._members:
            raise Exception("{} is not in the aggregator".format(key))
        v, _ = self._members[key]
        return v

    def get_weight(self, key):
        if key not in self._members:
            raise Exception("{} is not in the aggregator".format(key))
        _, w = self._members[key]
        return w

    def pop(self, key):
        if key not in self._members:
            raise Exception("{} is not in the aggregator".format(key))
        v, w = self._members.pop(key)
        if w is None or w == 0:
            return v
        return v / w

    def items(self):
        for key, _ in self._members.items():
            yield key, self.get(key)

    def pop_all(self):
        return {key: self.pop(key) for key in list(self._members.keys())}

    def __contains__(self, key):
        return key in self._members
