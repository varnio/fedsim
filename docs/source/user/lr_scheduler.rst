.. _custom_lrsch:

Guide to learning rate schedulers
=================================

`fedsim-cli fed-learn` accepts 3 scheduler objects.

* **lr-scheduler:** learning rate scheduler for server optimizer.
* **local-lr-scheduler:** learning rate scheduler for client optimizer.
* **r2r-local-lr-scheduler:** schedules the initial learning rate that is delivered to the clients of each round.

These arguments are passed to instances of the centralized FL algorithms.

.. note::
    Choose learning rate schedulers from ``torch.optim.lr_scheduler`` documented at `Lr Schedulers Page`_ or define a learning rate scheduler class that has the common methods (``step``, ``get_last_lr``, etc.).

.. _Lr Schedulers Page: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts

.. note::
    For now ``fedsim-cli`` does not support the learning rate schedulers that require another object in their constructor (such as ``LambdaLR``) or a dynamic value in their step function (``ReduceLROnPlateau``).
    To implement one with similar functionality, you can implement one and assign it to ``self.r2r_local_lr_scheduler`` inside the constructor of your custom algorithm (after calling super).


Custom Learning Rate Scheduler
------------------------------

Any custom learning rate scheduler class should implement the common methods of torch optim lr schedulers.


Integration with fedsim-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom lr scheduler by the provided cli tool, you can define it in a python file and pass its path to ``--lr-scheduler`` or ``--local-lr-scheduler`` or ``r2r-local-lr-scheduler`` option (without .py) followed by column and name of the lr scheduler definition (class or method).
For example, if you have score ``CustomLRS`` stored in a ``foo/bar/my_custom_lr_scheduler.py``, you can pass ``--lr-scheduler foo/bar/my_custom_lr_scheduler:CustomLRS`` for setting global lr scheduler or ``--local-lr-scheduler foo/bar/my_custom_lr_scheduler:CustomLRS`` for setting the local lr scheduler or ``--r22-local-lr-scheduler foo/bar/my_custom_lr_scheduler:CustomLRS`` for setting the round to round lr scheduler.
The latter determines the initial learning rate of the local optimizer at each round.

.. note::

    Arguments of constructor of any lr scheduler could be given in ``arg:value`` format following its name (or `path` if a local file is provided). Examples:

    .. code-block:: bash

        fedsim-cli fed-learn --lr-scheduler StepLR step_size:200 gamma:0.5 ...

    .. code-block:: bash

        fedsim-cli fed-learn --local-lr-scheduler foo/bar/my_custom_lr_scheduler:CustomLRS step_size:10 beta:0.1 ...
