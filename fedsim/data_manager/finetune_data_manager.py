from typing import Iterable, Dict
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import numpy as np
from tqdm import tqdm
import torchvision

from fedsim.data_manager.basic_data_manager import BasicDataManager


class FineTuneDataManager(BasicDataManager):

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
        valid_portion=0.3,
        stratified=True,
        *args,
        **kwargs,
    ):
        """A data manager for partitioning the data to be used in fine tuning.
        Currecntly three rules of partitioning are supported:
        
        - iid: 
            same label distribution among clients. sample balance determines 
            quota of each client samples from a lognorm distribution. 
        - dir: 
            Dirichlete distribution with concentration parameter given by 
            label_balance determines label balance of each client. 
            sample balance determines quota of each client samples from a
            lognorm distribution. 
        - exclusive: 
            samples corresponding to each label are randomly splitted to 
            k clients where k = total_sample_size * label_balance. 
            sample_balance determines the way this split happens (quota).
            This rule also is know as "shards splitting".

        Args:
            root (str): root dir of the dataset to partition
            dataset (str): name of the dataset
            num_clients (int): number of partitions or clients
            rule (str): rule of partitioning
            sample_balance (float): balance of number of samples among clients
            label_balance (float): balance of the labels on each clietns
            seed (int): random seed of partitioning
            save_path (str, optional): path to save partitioned indices.
        """
        super(FineTuneDataManager, self).__init__(
            root,
            dataset,
            num_clients,
            rule,
            sample_balance,
            label_balance,
            seed,
            save_path,
        )

    def make_datasets(self, dataset_name, root):
        if dataset_name == 'mnist':
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
            local_datasets = dict(train=MNIST(root=root,
                                              download=True,
                                              train=True,
                                              transform=train_transform))
            test_transform = train_transform
            global_datasets = dict(test=MNIST(root=root,
                                              download=True,
                                              train=False,
                                              transform=test_transform))
            return local_datasets, global_datasets

        if dataset_name == 'cifar10' or dataset_name == 'cifar100':
            dst_class = CIFAR10 if dataset_name == 'cifar10' else CIFAR100
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomCrop(24),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=(0.5, 1.5),
                                                   contrast=(0.5, 1.5)),
            ])
            local_datasets = dict(train=dst_class(root=root,
                                                  download=True,
                                                  train=True,
                                                  transform=train_transform))
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.CenterCrop(24),
            ])
            global_datasets = dict(test=dst_class(root=root,
                                                  download=True,
                                                  train=False,
                                                  transform=test_transform))
            return local_datasets, global_datasets

        raise NotImplementedError

    def partition_local_data(self, datasets, num_clients, rule, sample_balance,
                             label_balance):
        indices_dict = dict()
        for key, dataset in datasets.items():
            targets = np.array(dataset.targets)
            all_sample_count = len(targets)
            num_classes = np.unique(targets)
            # the special case of exclusive rule:
            if self.rule == 'exclusive':
                # TODO: implement this
                raise NotImplementedError
            #     # get number of samples per label
            #     label_counts = [(targets==i).sum() for i in range(num_classes)]
            #     for label, label_count in enumerate(label_counts):
            #         # randomly select k clients
            #         # determine the quota for each client from a lognorm
            #         # reassign the
            # *********************************************************
            # determine sample quota for each client

            sample_per_client = all_sample_count // num_clients
            if sample_balance != 0:
                # Draw from lognormal distribution
                client_quota = (np.random.lognormal(
                    mean=np.log(sample_per_client),
                    sigma=sample_balance,
                    size=num_clients))
                quota_sum = np.sum(client_quota)
                client_quota = (client_quota / quota_sum *
                                all_sample_count).astype(int)
                diff = quota_sum - all_sample_count

                # Add/Sub the excess number starting from first client
                if diff != 0:
                    for clnt_i in range(num_clients):
                        if client_quota[clnt_i] > diff:
                            client_quota[clnt_i] -= diff
                            break
            else:
                client_quota = np.ones(num_clients, dtype=int) *\
                    sample_per_client

            indices = [
                np.zeros(client_quota[client], dtype=int) \
                    for client in range(num_clients)
             ]
            # *********************************************************
            if rule == 'dir':
                # Dirichlet partitioning rule
                cls_priors = np.random.dirichlet(alpha=[label_balance] *
                                                 num_classes,
                                                 size=num_clients)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [
                    np.where(targets == i)[0] for i in range(num_classes)
                ]
                cls_amount = np.array(
                    [len(idx_list[i]) for i in range(num_clients)])

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
                    # scale the prior up so the positive values sum to
                    # 1 again
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
                            cls_label][cls_amount[cls_label]]

                        break
                    pbar.update(1)

                pbar.close()
            # *********************************************************
            elif self.rule == 'iid':
                clnt_quota_cum_sum = np.concatenate(
                    ([0], np.cumsum(client_quota)))
                for client_index in range(num_clients):
                    indices[client_index] = np.arange(
                        clnt_quota_cum_sum[client_index],
                        clnt_quota_cum_sum[client_index + 1])
            else:
                raise NotImplementedError

            indices_dict[key] = indices

        return indices_dict
