.. _custom_dm:

Custom DataManager
==================

Any custome DataManager class should inherit from ``fedsim.data_manager.data_manager.DataManager`` (or its children) and implement its abstract methods.


A Simple Custom DataManager
---------------------------

.. code-block:: python

   from fedsim.data_manager.data_manager import DataManager

   class CustomDataManager(DataManager)
       def __init__(self, root, other_arg, ...):
           self.other_arg = other_arg
           # note that super should be called at the end of init \
           # because the abstract classes are called in its __init__
           super(BasicDataManager, self).__init__(root, seed, save_path=save_path)

       def make_datasets(self, root: str) -> Iterable[Dict[str, object]]:
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


Integration with included cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom data manager in the provided cli tool, you can place your class in a file under ``fedsim/data_manager``. Then, call it using option ``--data-manager``. To deliver arguments to the ``__init__`` method of your custom data manager, you can pass options in form of ``--d-<arg-name>`` where ``<arg-name>`` is the argument. Example

.. code-block:: bash

   fedsim fed-learn --data-manager CustomDataManager --d-other_arg <other_arg_value> ...