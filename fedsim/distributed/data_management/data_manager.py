r"""
Data Manager
------------

"""
import os
import pickle
import random
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple

# import libraries with random generator to set seed
import numpy as np
from torch.utils.data import Dataset

from .utils import Subset


class DataManager(object):
    r"""DataManager base class.
    Any other Data Manager is inherited from this class. There are
    four abstract class methods that child classes should implement:
    get_identifiers, make_datasets, make_transforms, partition_local_data.

    .. warning::
        when inheritted, super should be called at the end of the constructor
        because the abstract classes are called in super's constructor!

    Args:
        root (str): root dir of the dataset to partition
        seed (int): random seed of partitioning
        save_dir (str, optional): path to save partitioned indices.
    """

    def __init__(
        self,
        root,
        seed,
        save_dir=None,
    ):
        self.root = root
        self.seed = seed
        self.save_dir = root if save_dir is None else save_dir

        # *********************************************************************
        # define class vars

        # {'<split_name>': <dataset_obj>}
        self.local_data: Optional[Dict[str, Dataset]] = None
        self.global_data: Optional[Dict[str, Dataset]] = None

        self.global_transforms = None
        self.local_transforms = None

        # {'<split_name>': [<client index>:[<sample_index>,],]}
        self._local_parition_indices: Optional[Dict[Iterable[Iterable[int]]]] = None

        # set random seed for partitioning
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # prepare stuff
        self._make_transforms()
        self._make_datasets()
        self._partition_local_data()
        self._partition_global_data()

    def _make_transforms(self) -> None:
        (
            self.local_transforms,
            self.global_transforms,
        ) = self.make_transforms()

    def _make_datasets(self) -> None:
        self.local_data, self.global_data = self.make_datasets(
            self.root,
        )

    def _partition_local_data(self) -> None:
        if self.local_data is None:
            raise Exception(
                "call a make_datasets that returns a dict of datasets first!"
            )
        name = self.get_partitioning_name()
        if os.path.exists(os.path.join(self.save_dir, name + "_local.pkl")):
            with open(os.path.join(self.save_dir, name + "_local.pkl"), "rb") as rfile:
                self._local_parition_indices = pickle.load(rfile)
        else:
            self._local_parition_indices = self.partition_local_data(self.local_data)
            # save on disk for later usage

            # create directories if not existing
            os.makedirs(os.path.join(self.save_dir), exist_ok=True)
            with open(os.path.join(self.save_dir, name + "_local.pkl"), "wb") as wfile:
                pickle.dump(self._local_parition_indices, wfile)

    def _partition_global_data(self) -> None:
        if self.global_data is not None:
            name = self.get_partitioning_name()
        if os.path.exists(os.path.join(self.save_dir, name + "_global.pkl")):
            with open(os.path.join(self.save_dir, name + "_global.pkl"), "rb") as rfile:
                self._global_parition_indices = pickle.load(rfile)
        else:
            self._global_parition_indices = self.partition_global_data(self.local_data)
            # save on disk for later usage

            # create directories if not existing
            os.makedirs(os.path.join(self.save_dir), exist_ok=True)
            with open(os.path.join(self.save_dir, name + "_global.pkl"), "wb") as wfile:
                pickle.dump(self._global_parition_indices, wfile)

    # *************************************************************************
    # to be called by user
    def get_local_dataset(self, id: int) -> Dict[str, Dataset]:
        """returns the local dataset corresponding to a given partition id

        Args:
            id (int): partition id

        Returns:
            Dict[str, Dataset]: a mapping of split_name: dataset
        """
        ans = dict()
        for key, val in self._local_parition_indices.items():
            if len(val) > 0:
                ans[key] = Subset(
                    self.local_data,
                    val[id],
                    transform=self.local_transforms[key],
                )
        return ans

    def get_group_dataset(self, ids: Iterable[int]) -> Dict[str, Dataset]:
        """returns the local dataset corresponding to a group of given partition ids

        Args:
            ids (Iterable[int]): a list or tuple of partition ids

        Returns:
            Dict[str, Dataset]: a mapping of split_name: dataset
        """
        ans = dict()
        for key, val in self._local_parition_indices.items():
            group_split_idxs = [i for id in ids for i in val[id]]
            if len(group_split_idxs) > 0:
                ans[key] = Subset(
                    self.local_data,
                    group_split_idxs,
                    transform=self.local_transforms[key],
                )
        return ans

    def get_oracle_dataset(self) -> Dict[str, Dataset]:
        """returns all of the local datasets stacked up.

        Returns:
            Dict[str, Dataset]: Oracle dataset for each split
        """
        return self.get_group_dataset(
            ids=range(len(self._local_parition_indices["train"]))
        )

    def get_global_dataset(self) -> Dict[str, Dataset]:
        """returns the global dataset

        Returns:
            Dict[str, Dataset]: global dataset for each split
        """
        ans = dict()
        for key, val in self._global_parition_indices.items():
            if len(val) > 0:
                ans[key] = Subset(
                    self.local_data,
                    val,
                    transform=self.global_transforms[key],
                )
        return ans

    def get_partitioning_name(self) -> str:
        """returns unique name of the DataManager instance.
        .. note::
        This method can help store and retrieval of the partitioning indices, so
        the experiments could reproduced on a machine.

        Returns:
            str: a unique name for the DataManager instance.
        """
        identifiers = self.get_identifiers()
        name = "_".join(identifiers)
        if self.seed is not None:
            name += "_seed{}".format(self.seed)
        return name

    def get_local_splits_names(self):
        """returns name of the local splits (train, test, etc.)

        Returns:
            List[str]: list of local split names
        """
        return list(self._local_parition_indices.keys())

    def get_global_splits_names(self):
        """returns name of the global splits (train, test, etc.)

        Returns:
            List[str]: list of global split names
        """
        return list(self._global_parition_indices.keys())

    # *************************************************************************
    # to implement by child
    def make_datasets(self, root: str) -> Tuple[object, object]:
        """makes and returns local and global dataset objects. The created datasets do
        not need a transform as recompiled datasets with separately provided transforms
        on the fly.

        Args:
            dataset_name (str): name of the dataset.
            root (str): directory to download and manipulate data.

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes

        Returns:
            Tuple[object, object]: local and global dataset
        """
        raise NotImplementedError

    def make_transforms(self) -> Tuple[object, object]:
        """make and return the dataset trasformations for local and global split.

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes
        Returns:
            Tuple[Dict[str, Callable], Dict[str, Callable]]: tuple of two dictionaries,
                first, the local transform mapping and second the global transform
                mapping.
        """
        raise NotImplementedError

    def partition_local_data(
        self,
        dataset: object,
    ) -> Dict[str, Iterable[Iterable[int]]]:
        """partitions local data indices into client-indexed Iterable.

        Args:
            dataset (object): local dataset

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes

        Returns:
            Dict[str, Iterable[Iterable[int]]]:
                dictionary of {split:client-indexed iterables of example indices}.
        """
        raise NotImplementedError

    def partition_global_data(
        self,
        dataset: object,
    ) -> Dict[str, Iterable[int]]:
        """partitions global data indices into splits (e.g., train, test, ...).

        Args:
            dataset (object): global dataset

        Returns:
            Dict[str, Iterable[int]]:
                dictionary of {split:example indices of global dataset}.
        """
        raise NotImplementedError

    def get_identifiers(self) -> Sequence[str]:
        """Returns identifiers to be used for saving the partition info.

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes

        Returns:
            Sequence[str]: a sequence of str identifing class instance
        """
        raise NotImplementedError
