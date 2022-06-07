import os
import pickle
from typing import Iterable, Dict, Optional, Sequence
from torch.utils.data import Dataset
from torch.utils.data import Subset

# import libraries with random generator to set seed
import numpy as np
import random


class DataManager(object):

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

        # {'<split_name>': [<client index>:[<sample_index>,],]}
        self._local_parition_indices: Optional[Dict[Iterable[
            Iterable[int]]]] = None

        # set random seed for partitioning
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # prepare stuff
        self._make_datasets()
        self._partition_local_data()

    def _make_datasets(self) -> None:
        self.local_data, self.global_data = self.make_datasets(self.root)

    def _partition_local_data(self) -> None:
        if self.local_data is None:
            raise Exception(
                "call a make_datasets that returns a dict of datasets first!")
        name = self.get_partitioning_name()
        if os.path.exists('{}/{}'.format(self.save_dir, name)):
            with open(os.path.join(self.save_dir, name + '.pkl'),
                      'rb') as rfile:
                self._local_parition_indices = pickle.load(rfile)
        else:

            self._local_parition_indices = self.partition_local_data(
                self.local_data, )
            # save on disk for later usage

            # create directories if not existing
            os.makedirs(os.path.join(self.save_dir), exist_ok=True)
            with open(os.path.join(self.save_dir, name + '.pkl'),
                      'wb') as wfile:
                pickle.dump(self._local_parition_indices, wfile)

    # *************************************************************************
    # to call by user
    def get_local_dataset(self, id: int) -> Dict[str, Dataset]:
        return {
            key: Subset(value, self._local_parition_indices[key][id])
            for key, value in self.local_data.items()
        }

    def get_group_dataset(self, ids: Iterable[int]) -> Dict[str, Dataset]:
        return {
            key:
                Subset(
                    value,
                    [
                        i for id in ids \
                            for i in self._local_parition_indices[key][id]
                    ]
                )
                for key, value in self.local_data.items()
        }

    def get_oracle_dataset(self) -> Dict[str, Dataset]:
        return {key: value for key, value in self.local_data.items()}

    def get_global_dataset(self) -> Dict[str, Dataset]:
        return self.global_data

    def get_partitioning_name(self) -> str:
        identifiers = self.get_identifiers()
        name = '_'.join(identifiers)
        if self.seed is not None:
            name += '_seed{}'.format(self.seed)
        return name

    # *************************************************************************
    # to implement by child
    def make_datasets(
        self,
        root: str,
    ) -> Iterable[Dict[str, object]]:
        """Abstract method to be implemented by child class.

        Args:
            dataset_name (str): name of the dataset.
            root (str): directory to download and manipulate data.
            save_path (str): directory to store the data after partitioning.

        Raises:
            NotImplementedError: if the dataset_name is not defined

        Returns:
            Iterable[Dict[str, object]]: dict of local datasets [split:dataset]
                                         followed by global ones.
        """
        raise NotImplementedError

    def partition_local_data(
        self,
        datasets: Dict[str, object],
    ) -> Dict[str, Iterable[Iterable[int]]]:
        raise NotImplementedError

    def get_identifiers(self) -> Sequence[str]:
        """ Returns identifiers 
            to be used for saving the partition info.

        Raises:
            NotImplementedError: this abstract method should be 
                implemented by child classes

        Returns:
            Sequence[str]: a sequence of str identifing class instance 
        """
        raise NotImplementedError
