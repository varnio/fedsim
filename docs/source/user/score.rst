.. _custom_score:

Guide to scores
===============


Custom scores
-------------

Any custom score class should inherit from ``fedsim.scores.Score`` (or its children) and implement its abstract methods.


Integration with fedsim-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom score by the provided cli tool, you can define it in a python file and pass its path to ``--global-score`` or ``--local-score`` option (without .py) followed by column and name of the score definition (class or method).
For example, if you have score ``CustomScore`` stored in a ``foo/bar/my_custom_score.py``, you can pass ``--global-score foo/bar/my_custom_score:CustomScore`` for setting global optimizer or ``--local-score foo/bar/my_custom_score:CustomScore`` for setting the local score.

.. note::

    Arguments of constructor of any score could be given in ``arg:value`` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --global-score Accuracy log_freq:20 split:test ...

    .. code-block:: bash

        fedsim-cli fed-learn --local-score foo/bar/my_custom_sore:CustomScore log_freq:30 split:train ...


.. note::
    scores can be passed to ``--criterion`` option the same way, however, if the selected score class is not differentiable an error may be raised (if necessary).
