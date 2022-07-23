import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import MNIST
from tqdm import tqdm

from .data_manager import DataManager


class BasicDataManager(DataManager):
    r"""A basic data manager for partitioning the data. Currecntly three
    rules of partitioning are supported:

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
        local_test_portion (float): portion of local test set from trian
        seed (int): random seed of partitioning
        save_dir (str, optional): dir to save partitioned indices.
    """

    def __init__(
        self,
        root,
        dataset="mnist",
        num_partitions=500,
        rule="iid",
        sample_balance=0.0,
        label_balance=1.0,
        local_test_portion=0.0,
        seed=10,
        save_dir=None,
        *args,
        **kwargs,
    ):
        """A basic data manager for partitioning the data. Currecntly three
        rules of partitioning are supported:

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
            local_test_portion (float): portion of local test set from trian
            seed (int): random seed of partitioning
            save_dir (str, optional): path to save partitioned indices.
        """
        self.dataset_name = dataset
        self.num_partitions = num_partitions
        self.rule = rule
        self.sample_balance = sample_balance
        self.label_balance = label_balance
        self.local_test_portion = local_test_portion

        # super should be called at the end because abstract classes are
        # called in its __init__
        super(BasicDataManager, self).__init__(
            root,
            seed,
            save_dir=save_dir,
        )

    def make_datasets(self, root, global_transforms=None):
        if self.dataset_name == "mnist":
            local_dset = MNIST(root, download=True, train=True, transform=None)
            global_dset = MNIST(root, download=True, train=True, transform=None)

        elif self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            dst_class = CIFAR10 if self.dataset_name == "cifar10" else CIFAR100

            local_dset = dst_class(root=root, download=True, train=True, transform=None)
            global_dset = dst_class(
                root=root,
                download=True,
                train=False,
                transform=global_transforms["test"],
            )
        else:
            raise NotImplementedError
        return local_dset, global_dset

    def make_transforms(self):
        if self.dataset_name == "mnist":
            train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
            test_transform = train_transform
        if self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.RandomCrop(24),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ColorJitter(
                        brightness=(0.5, 1.5), contrast=(0.5, 1.5)
                    ),
                ]
            )
            test_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.CenterCrop(24),
                ]
            )
        return train_transform, test_transform

    def partition_local_data(self, dataset):
        n = self.num_partitions

        targets = np.array(dataset.targets)
        all_sample_count = len(targets)
        num_classes = len(np.unique(targets))
        # the special case of exclusive rule:
        if self.rule == "exclusive":
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

        sample_per_client = all_sample_count // n
        if self.sample_balance != 0:
            # Draw from lognormal distribution
            client_quota = np.random.lognormal(
                mean=np.log(sample_per_client),
                sigma=self.sample_balance,
                size=n,
            )
            quota_sum = np.sum(client_quota)
            client_quota = (client_quota / quota_sum * all_sample_count).astype(int)
            diff = quota_sum - all_sample_count

            # Add/Sub the excess number starting from first client
            if diff != 0:
                for clnt_i in range(n):
                    if client_quota[clnt_i] > diff:
                        client_quota[clnt_i] -= diff
                        break
        else:
            client_quota = np.ones(n, dtype=int) * sample_per_client

        indices = [np.zeros(client_quota[client], dtype=int) for client in range(n)]
        # *********************************************************
        if self.rule == "dir":
            # Dirichlet partitioning rule
            cls_priors = np.random.dirichlet(
                alpha=[self.label_balance] * num_classes, size=n
            )
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
            cls_amount = np.array([len(idx_list[i]) for i in range(num_classes)])

            print("partitionig")
            pbar = tqdm(total=np.sum(client_quota))
            while np.sum(client_quota) != 0:
                curr_clnt = np.random.randint(n)
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
                    indices[curr_clnt][client_quota[curr_clnt]] = idx_list[cls_label][
                        cls_amount[cls_label]
                    ]

                    break
                pbar.update(1)

            pbar.close()
        # *********************************************************
        elif self.rule == "iid":
            clnt_quota_cum_sum = np.concatenate(([0], np.cumsum(client_quota)))
            for client_index in range(n):
                indices[client_index] = np.arange(
                    clnt_quota_cum_sum[client_index],
                    clnt_quota_cum_sum[client_index + 1],
                )
        else:
            raise NotImplementedError

        ts_portion = self.local_test_portion
        if ts_portion > 0:
            new_indices = dict(train=[], test=[])
            for client_indices in indices:
                train_idxs, test_idxs = train_test_split(
                    client_indices, test_size=ts_portion
                )
                new_indices["train"].append(train_idxs)
                new_indices["test"].append(test_idxs)
        else:
            new_indices = dict(train=indices)
        return new_indices

    def get_identifiers(self):
        identifiers = [
            self.dataset_name,
            str(self.num_partitions),
            self.rule,
        ]
        if self.rule == "dir":
            identifiers.append(str(self.label_balance))
        if self.sample_balance == 0:
            identifiers.append("balanced")
        else:
            identifiers.append("unbalanced")
        if self.local_test_portion > 0:
            identifiers.append("ts_{}".format(self.local_test_portion))
        return identifiers
