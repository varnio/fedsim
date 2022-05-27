from typing import Iterable, Dict
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm

from data_manager.base_data_manager import BaseDataManager


class FedDynDataManager(BaseDataManager):
    def __init__(
        self, root: str, dataset: str, num_clients: int, rule: str,
        sample_balance: float, label_balance: float, seed: int,
        save_path: str = None
        ) -> None:
        super().__init__(
            root, dataset, num_clients, rule, sample_balance, label_balance, 
            label_balance, seed, save_path
            )
    
    # to implement by child
    def make_datasets(
        self, dataset_name: str, root: str, save_path: str,
        ) -> Iterable[Dict[str, object]]:
        if dataset_name == 'MNIST':
            return {'train': MNIST(root=root, download=True, train=True)},\
                   {'test': MNIST(root=root, download=True, train=False)}
        raise NotImplementedError

    def partition_local_data(
        self, datasets: Dict[str, object], num_clients: int, rule: str, 
        sample_balance: float, label_balance: float,
        ) -> Dict[Iterable[Iterable[int]]]:
        
        targets = np.array(dataset.targets)
        all_sample_count = len(targets)
        num_classes = np.unique(targets)

        indices_dict= dict()
        for key, dataset in datasets:
            # *********************************************************************
            # determine sample quota for each client

            sample_per_client = all_sample_count // num_clients
            if sample_balance != 0:
                # Draw from lognormal distribution
                client_quota = (
                    np.random.lognormal(
                        mean=np.log(sample_per_client), 
                        sigma=sample_balance, size=num_clients
                        )
                    )
                quota_sum = np.sum(client_quota)
                client_quota = (
                    client_quota / quota_sum * all_sample_count
                    ).astype(int)
                diff = quota_sum - all_sample_count

                # Add/Subtract the excess number starting from first client
                if diff != 0:
                    for clnt_i in range(num_clients):
                        if client_quota[clnt_i] > diff:
                            client_quota[clnt_i] -= diff
                            break

            else:
                client_quota = np.ones(
                    num_clients, dtype=int
                    ) * sample_per_client

            indices = [
                np.zeros(client_quota[client], dtype=int) \
                    for client in range(num_clients)
             ]
            # *****************************************************************
            if rule == 'dir':
                # Dirichlet partitioning rule
                cls_priors = np.random.dirichlet(
                    alpha=[ label_balance ] * num_classes, size=num_clients
                    )
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [
                    np.where(targets == i)[0] for i in range(num_classes)
                    ]
                cls_amount = np.array(
                    [len(idx_list[i]) for i in range(num_clients)]
                    )

                print('partitionig')
                pbar = tqdm(total=np.sum(client_quota))
                while np.sum(client_quota) != 0:
                    curr_clnt = np.random.randint(num_clients)
                    # If current node is full resample a client
                    if client_quota[curr_clnt] <= 0:
                        continue
                    client_quota[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    # exclude the classes that have ran out of examples
                    curr_prior[cls_amount <= 0] = -1
                    # scale the prior up so the positive values sum to 1 again
                    cpp = curr_prior[curr_prior > 0]
                    cpp /= cpp.sum()
                    curr_prior[curr_prior > 0] = cpp

                    while True:
                        if (curr_prior > 0).sum() < 1:
                            raise Exception("choose another seed")
                        if (curr_prior > 0).sum() == 1:
                            cls_label = curr_prior.argmax()
                        else:

                            uu = np.random.uniform()
                            cls_label = np.argmax(uu <= curr_prior)
                        # Redraw class label if out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        indices[curr_clnt][client_quota[curr_clnt]] = idx_list[
                            cls_label][cls_amount[cls_label]
                            ]

                        break
                    pbar.update(1)  
                    
                pbar.close()
            # *****************************************************************
            elif self.rule == 'iid':
                clnt_quota_cum_sum = np.concatenate(
                    ([0], np.cumsum(client_quota))
                    )
                for client_index in range(num_clients):
                    indices[client_index] = np.arange(
                        clnt_quota_cum_sum[client_index], 
                        clnt_quota_cum_sum[client_index + 1]
                        )

            indices_dict[key] = np.asarray(indices)

        return indices_dict    
