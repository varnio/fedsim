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
        *args,
        **kwargs,
    ):
        self.root = root
        self.seed = seed
        self.save_dir = root if save_dir is None else save_dir

        # *********************************************************************
        # define class vars

        # {'<split_name>': <dataset_obj>}
        self.local_data: Optional[Dict[str, Dataset]] = None
        self.global_data: Optional[Dict[str, Dataset]] = None

        self.train_transforms = None
        self.test_transforms = None

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

    def _make_transforms(self) -> None:
        (
            self.train_transforms,
            self.test_transforms,
        ) = self.make_transforms()

    def _make_datasets(self) -> None:
        if self.test_transforms is None:
            raise Exception("call make_tranforms() before make_datasets()")
        self.local_data, self.global_data = self.make_datasets(
            self.root,
            dict(train=self.train_transforms, test=self.test_transforms),
        )
        if self.global_data is not None:
            self.global_data = Subset(
                self.global_data,
                indices=-1,
                transform=self.test_transforms,
            )

    def _partition_local_data(self) -> None:
        if self.local_data is None:
            raise Exception(
                "call a make_datasets that returns a dict of datasets first!"
            )
        name = self.get_partitioning_name()
        if os.path.exists(os.path.join(self.save_dir, name + ".pkl")):
            with open(os.path.join(self.save_dir, name + ".pkl"), "rb") as rfile:
                self._local_parition_indices = pickle.load(rfile)
        else:
            self._local_parition_indices = self.partition_local_data(self.local_data)
            # save on disk for later usage

            # create directories if not existing
            os.makedirs(os.path.join(self.save_dir), exist_ok=True)
            with open(os.path.join(self.save_dir, name + ".pkl"), "wb") as wfile:
                pickle.dump(self._local_parition_indices, wfile)

    # *************************************************************************
    # to call by user
    def get_local_dataset(self, id: int) -> Dict[str, Dataset]:
        tr_idxs = self._local_parition_indices["train"]
        tr_dset = Subset(
            self.local_data,
            tr_idxs[id],
            transform=self.train_transforms,
        )
        if "test" in self._local_parition_indices:
            ts_idxs = self._local_parition_indices["test"]
            if len(tr_idxs) > 0:
                ts_dset = Subset(
                    self.local_data,
                    ts_idxs[id],
                    transform=self.test_transforms,
                )
                return dict(train=tr_dset, test=ts_dset)
        return dict(train=tr_dset)

    def get_group_dataset(self, ids: Iterable[int]) -> Dict[str, Dataset]:
        tr_idxs = self._local_parition_indices["train"]
        group_tr_idxs = [i for id in ids for i in tr_idxs[id]]
        tr_dset = Subset(
            self.local_data,
            group_tr_idxs,
            transform=self.train_transforms,
        )
        if "test" in self._local_parition_indices:
            ts_idxs = self._local_parition_indices["test"]
            if len(tr_idxs) > 0:
                group_ts_idxs = [i for id in ids for i in ts_idxs[id]]
                ts_dset = Subset(
                    self.local_data,
                    group_ts_idxs,
                    transform=self.test_transforms,
                )
                return dict(train=tr_dset, test=ts_dset)
        return dict(train=tr_dset)

    def get_oracle_dataset(self) -> Dict[str, Dataset]:
        return self.get_group_dataset(
            ids=range(len(self._local_parition_indices["train"]))
        )

    def get_global_dataset(self) -> Dict[str, Dataset]:
        return dict(test=self.global_data)

    def get_partitioning_name(self) -> str:
        identifiers = self.get_identifiers()
        name = "_".join(identifiers)
        if self.seed is not None:
            name += "_seed{}".format(self.seed)
        return name

    def get_local_splits_names(self):
        return list(self._local_parition_indices.keys())

    # *************************************************************************
    # to implement by child
    def make_datasets(
        self, root: str, global_transforms: Dict[str, object]
    ) -> Tuple[object, object]:
        """makes and returns local and global dataset objects. The local
            datasets do not need a transform as recompiled datasets from
            indices already use transforms as they are requested.

        Args:
            dataset_name (str): name of the dataset.
            root (str): directory to download and manipulate data.
            global_transforms (Dict[str, object]): transforms for global dset

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes

        Returns:
            Tuple[object, object]: local and global dataset
        """
        raise NotImplementedError

    def make_transforms(self) -> Tuple[object, object]:
        """makes and returns train and inference transforms.

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes

        Returns:
            Tuple[object, object]: train and inference transforms
        """
        raise NotImplementedError

    def partition_local_data(
        self,
        dataset: object,
    ) -> Dict[str, Iterable[Iterable[int]]]:
        """partitions local data indices into client index Iterable.

        Args:
            dataset (object): local dataset

        Raises:
            NotImplementedError: this abstract method should be
                implemented by child classes

        Returns:
            Dict[str, Iterable[Iterable[int]]]:
                {'train': tr_indices, 'test': ts_indices}
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
