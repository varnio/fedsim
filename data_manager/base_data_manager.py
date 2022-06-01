import os
import pickle
from typing import Iterable, Dict, Optional
from torch import seed
from torch.utils.data import Dataset
from torch.utils.data import Subset

# import libraries with random generator to set seed
import numpy as np
import random


class BaseDataManager(object):
    def __init__(
        self,
        root,
        dataset,
        num_clients,
        rule,
        sample_balance,
        label_balance,
        seed, 
        save_path=None,
    ):
        self.root = root
        self.dataset_name = dataset
        self.num_clients = num_clients
        self.rule = rule
        self.sample_balance = sample_balance
        self.label_balance = label_balance
        self.seed = seed
        self.save_path = save_path
        
        # *********************************************************************
        # define class vars

        # {'<split_name>': <dataset_obj>}
        self.local_data: Optional[Dict[str, Dataset]] = None
        self.global_data: Optional[Dict[str, Dataset]] = None
        
        # {'<split_name>': [<client index>:[<sample_index>,],]}
        self._local_parition_indices: Optional[
            Dict[Iterable[Iterable[int]]]
        ] =  None
        # prepare stuff
        self._make_datasets()
        self._partition_local_data()

    def get_partitioning_name(self) -> str:
        partitioning_name = '{}_{}_{}'.format(
            self.dataset_name, self.num_clients, self.rule
        )
        if self.rule == 'dir':
            partitioning_name += '_{}'.format(self.label_balance)
        if self.sample_balance == 0:
            partitioning_name += '_balanced'
        else:
            partitioning_name += '_unbalanced_{}'.format(self.sample_balance)
        if self.seed is not None:
            partitioning_name += '_seed_{}'.format(self.seed)
        # set random seed for partitioning
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        return partitioning_name

    def _make_datasets(self) -> None:
        self.local_data, self.global_data = self.make_datasets(
            self.dataset_name,
            self.root,
        )
    
    def _partition_local_data(self) -> None:
        if self.local_data is None:
            raise Exception(
                "call a make_datasets that returns a dict of datasets first!"
            )
        partitioning_name = self.get_partitioning_name()
        if os.path.exists(
            '{}/{}'.format(self.save_path, partitioning_name)
            ):
            with open(
                os.path.join(self.save_path, partitioning_name + '.pkl'), 'rb'
            ) as rfile:
                self._local_parition_indices = pickle.load(rfile)    
        else:

            self._local_parition_indices = self.partition_local_data(
                self.local_data, self.num_clients, self.rule, 
                self.sample_balance, self.label_balance,
            )
            # save on disk for later usage
            
            # create directories if not existing
            os.makedirs(
                os.path.join(self.save_path), exist_ok=True
            )
            with open(
                os.path.join(self.save_path, partitioning_name + '.pkl'), 'wb'
            ) as wfile:
                pickle.dump(self._local_parition_indices, wfile)

    # *************************************************************************
    # to call by user
    def get_local_dataset(self, id: int) -> Dict[str, Dataset]:
        return {
            key: 
                Subset(
                    value, self._local_parition_indices[key][id]
                ) for key, value in self.local_data.items()
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
        return {
            key: value for key, value in self.local_data.items()
        }
            
    def get_global_dataset(self) -> Dict[str, Dataset]:
        return self.global_data
    # *************************************************************************
    # to implement by child
    def make_datasets(
        self, 
        dataset_name: str,
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
        num_clients: int, 
        rule: str, 
        sample_balance: float,
        label_balance: float,
    ) -> Dict[str, Iterable[Iterable[int]]]:
        raise NotImplementedError
