.. _custom_mdl:

Custom Model
============

Any custom model class should inherit from ``torch.Module`` (or its children) and implement its abstract methods.


Integration with fedsim-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom model by the provided cli tool, you can place your class in a python and pass its path to `-m` or `--model` option (without .py) followed by column and name of the model.
For example, if you have model `CustomModel` stored in a `foo/bar/my_custom_model.py`, you can pass `--model foo/bar/my_custom_alg:CustomModel`.

.. note::

    Arguments of the **init** method of any model could be given in `arg:value` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --model cnn_mnist num_classes:8 ...

    .. code-block:: bash

        fedsim-cli fed-learn --model foo/bar/my_custom_alg:CustomModel num_classes:8 ...
