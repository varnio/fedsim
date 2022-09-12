.. _custom_opt:

Guide to optimziers
===================

Custom optimziers
-----------------

Any custom optimizer class should inherit from ``torch.optim.Optimizer`` (or its children) and implement its abstract methods.


Integration with fedsim-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom optimizer by the provided cli tool, you can define it in a python file and pass its path to ``--optimzier`` or ``--local-optimzier`` option (without .py) followed by column and name of the optimizer definition (class or method).
For example, if you have optimizer ``CustomOpt`` stored in a ``foo/bar/my_custom_opt.py``, you can pass ``--optimizer foo/bar/my_custom_opt:CustomOpt`` for setting global optimizer or ``--local-optimizer foo/bar/my_custom_opt:CustomOpt`` for setting the local optimizer.

.. note::

    Arguments of constructor of any optimzier could be given in ``arg:value`` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --optimzier SGD lr:0.1 weight_decay:0.001 ...

    .. code-block:: bash

        fedsim-cli fed-learn --local-optimizer foo/bar/my_custom_opt:CustomOpt lr:0.2 momentum:True ...
