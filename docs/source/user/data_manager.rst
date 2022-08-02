.. _custom_dm:

Custom DataManager
==================

Any custome DataManager class should inherit from ``fedsim.data_manager.data_manager.DataManager`` (or its children) and implement its abstract methods.


A Simple Custom DataManager
---------------------------

.. code-block:: python

   from fedsim.distributed.data_management import DataManager

   class CustomDataManager(DataManager)
       def __init__(self, root, other_arg, ...):
           self.other_arg = other_arg
           # note that super should be called at the end of init \
           # because the abstract classes are called in its __init__
           super(BasicDataManager, self).__init__(root, seed, save_dir=save_dir)

       def make_datasets(self, root: str) -> Iterable[Dict[str, object]]:
           """Abstract method to be implemented by child class.

           Args:
               dataset_name (str): name of the dataset.
               root (str): directory to download and manipulate data.
               save_dir (str): directory to store the data after partitioning.

           Raises:
               NotImplementedError: if the dataset_name is not defined

           Returns:
               Iterable[Dict[str, object]]: dict of local datasets [split:dataset]
                                            followed by global ones.
           """
           raise NotImplementedError


       def partition_local_data(self, datasets: Dict[str, object]) -> Dict[str, Iterable[Iterable[int]]]:
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


Integration with fedsim-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom data-manager by the provided cli tool, you can place your class in a python file and pass its path to `-a` or `--data-manager` option (without .py) followed by column and name of the data-manager.
For example, if you have data-manager `DataManager` stored in `foo/bar/my_custom_dm.py`, you can pass `--data-manager foo/bar/my_custom_dm:DataManager`.

.. note::

    Arguments of the **init** method of any data-manager could be given in `arg:value` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --data-manager BasicDataManager num_clients:1100 ...

    .. code-block:: bash

        fedsim-cli fed-learn --data-manager foo/bar/my_custom_dm:DataManager arg1:value ...
