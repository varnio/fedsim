r"""
Storage
-------
"""


class Storage(object):
    r"""storage class to save and retrieve objects."""

    def __init__(self) -> None:
        self._storage = dict()

    def write(self, key, obj):
        r"""writes to the storage.

        Args:
            key (Hashable): key to access the object in future retrievals
            obj (Any): object to store
        """
        self._storage[key] = obj

    def read(self, key):
        """read from the storage.

        Args:
            key (Hashable): key to fetch the desired object.

        Returns:
            Any: the desired object. If key does not exist, None is returned.
        """
        return self._storage.get(key)

    def get_keys(self):
        """Fetches the keys of the objects written to the storage so far."""
        return self._storage.keys()
