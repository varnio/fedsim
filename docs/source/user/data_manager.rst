.. _custom_dm:

Guide to data manager
=====================

Provided with the simulator is a basic DataManager called ``BasicDataManager`` which for now supports the following datasets


* `MNIST <http://yann.lecun.com/exdb/mnist/>`_
* `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
* `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_

It supports the popular partitioning schemes (iid, Dirichlet distribution, unbalanced, etc.).


Custom DataManager
------------------

Any Custom data manager class should inherit from ``fedsim.data_manager.data_manager.DataManager`` (or its children) and implement its abstract methods.


DataManager Template
--------------------

.. code-block:: python

   from fedsim.distributed.data_management import DataManager

   class CustomDataManager(DataManager)
        def __init__(self, root, seed, save_dir=None, other_args="default value", ...):
            self.other_arg = other_arg
            """
            apply the changes required by the abstract methods here (before calling
            super's constructor).
            """
            super(BasicDataManager, self).__init__(root, seed, save_dir=save_dir)
            """
            apply the operation that assume the abstract methods are performed here
            (after calling super's constructor).
            """


        def make_datasets(self, root: str) -> Tuple[object, object]:
            """makes and returns local and global dataset objects. The created datasets do
            not need a transform as recompiled datasets with separately provided transforms
            on the fly (for vision datasets).

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

        def partition_local_data(self, datasets: Dict[str, object]) -> Dict[str, Iterable[Iterable[int]]]:
            """partitions local data indices into splits and within each split, partition in client-indexed Iterable.
            Return a dictionary of these splits (e.g., train, test, ...).

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
            """partitions global data indices into desired splits (e.g., train, test, ...).

            Args:
                dataset (object): global dataset

            Returns:
                Dict[str, Iterable[int]]:
                    dictionary of {split:example indices of global dataset}.
            """
            raise NotImplementedError

        def get_identifiers(self) -> Sequence[str]:
            """ Returns identifiers to be used for saving the partition info.
            A unique identifier for a unique setup ensures the credibility of comparing your experiments results.

            Raises:
                NotImplementedError: this abstract method should be
                    implemented by child classes

            Returns:
                Sequence[str]: a sequence of str identifing class instance
            """
            raise NotIm

.. note::
    scores can be passed to ``--criterion`` option the same way, however, if the selected score class is not differentiable an error may be raised (if necessary).plementedError

You can use `BasicDataManager as a working template <https://fedsim.varnio.com/en/latest/reference/fedsim.distributed.data_management.basic_data_manager.html>`_.


Integration with fedsim-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom data-manager into the provided cli tool, you can define it in a python file and pass its path to ``-a`` or ``--data-manager`` option (without .py) followed by colon and the definition of the data-manager (class or method).
For example, if you have data-manager ``DataManager`` stored in ``foo/bar/my_custom_dm.py``, you can pass ``--data-manager foo/bar/my_custom_dm:DataManager``.

.. note::

    Arguments of constructor of any data-manager could be given in ``arg:value`` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --data-manager BasicDataManager num_clients:1100 ...

    .. code-block:: bash

        fedsim-cli fed-learn --data-manager foo/bar/my_custom_dm:DataManager arg1:value ...
