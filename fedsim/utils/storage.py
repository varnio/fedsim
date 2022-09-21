r"""
Storage
-------
"""


class Storage(object):
    r"""storage class to save and retrieve objects."""

    def __init__(self) -> None:
        self._storage = dict()

    def write(
        self, key, obj, read_protected=False, write_protected=False, silent=False
    ):
        r"""writes to the storage.

        Args:
            key (Hashable): key to access the object in future retrievals
            obj (Any): object to store
            read_protected (bool): prints warning if in future key accessed by a read
                call. Defaults to False.
            write_protected (bool): print warning if in future key accessed by a write
                call. Defaults to False.
            silent (bool): if False and entry is write protected, a warning is printed.
                Defaults to False.
        """
        if key in self._storage:
            _, _, write_p = self._storage[key]
            if not silent and write_p:
                print(f"Warning: write protected entry {key} is over-written")
        self._storage[key] = (obj, read_protected, write_protected)

    def read(self, key, silent=False):
        """read from the storage.

        Args:
            key (Hashable): key to fetch the desired object.
            silent (bool): if False and entry is read protected, a warning is printed.
                Defaults to False.

        Returns:
            Any: the desired object. If key does not exist, None is returned.
        """
        res = self._storage.get(key)
        if res is None:
            return None
        obj, read_p, _ = res
        if not silent and read_p:
            print(f"Warning: read access to read-protected entry {key}")
        return obj

    def get_all_keys(self):
        """Fetches the keys of all the objects written to the storage so far including
        read protected ones.

        Returns:
            Iterable[str]: an iterable of the keys to the
        """
        return self._storage.keys()

    def get_keys(self):
        """Fetches the keys of the objects written to the storage so far.

        .. note::
            to get keys of all entries included read protected ones call
            ``get_all_keys`` instead.

        Returns:
            Iterable[str]: an iterable of the keys to the
        """
        keys = []
        for key, item in self._storage.items():
            r_p, _, _ = item
            if not r_p:
                keys.append(key)
        return keys

    def change_protection(
        self, key, read_protected=False, write_protected=False, silent=False
    ):
        """changes the protection policy of an entry

        Args:
            key (Hashable): key to the entry
            read_protected (bool, optional): read protection. Defaults to False.
            write_protected (bool, optional): write protection. Defaults to False.
            silent (bool): if False and and any protection changes, a warning is
                printed. Defaults to False.
        """
        if key not in self._storage:
            raise Exception(f"key {key} not in storage.")

        r_p, w_p, obj = self._storage[key]
        if not silent and r_p and not read_protected:
            print(f"read protection removed from storage entry {key}.")

        if not silent and w_p and not write_protected:
            print(f"write protection removed from storage entry {key}.")

        self._storage[key] = (read_protected, write_protected, obj)

    def get_protection_status(self, key):
        """fetches the protection status of an entry.

        Args:
            key (Hashable): key to the entry

        Returns:
            Tuple[bool, bool]: read and write protection status respectively.
        """
        if key not in self._storage:
            raise Exception(f"key {key} not in storage.")
        r_p, w_p, _ = self._storage[key]
        return r_p, w_p

    def remove(self, key, silent=False):
        """removes an entry from the storage.

        Args:
            key (Hashable): key to the entry.
            silent (bool, optional): if False and entry is write protected a warning is
                printed. Defaults to False.
        """
        if key not in self._storage:
            raise Exception(f"key {key} not in storage.")
        _, w_p, _ = self._storage[key]
        if not silent and w_p:
            print(f"write protected entry {key} is removed from the storage.")
        del self._storage[key]
