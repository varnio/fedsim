
************
Fine Tunning
************

The cli includes a fine-tunning tool. `fedsim-cli fed-tune` uses Bayesian optimization
provided by `skopt` to tune the hyper-parameters. Besides `skopt` argumetns, it accepts
all arguments that could be used by `fedsim-cli fed-learn`. The arguments values could
be defined as search spaces.

* To define a float range to tune use `Real` keyword as the argument value (e.g., `mu:Real:0-0.1`)
* To define an integer range to tune use `Integer` keyword as the argument value (e.g., `arg1:Integer:2-15`)
* To define a categorical range to tune use `Categorical` keyword as the argument value (e.g., `arg2:Categorical:uniform-normal-special`)

Examples

.. code-block:: bash

    fedsim-cli fed-tune --epochs 1 --n-clients 2 --client-sample-rate 0.5 -a AdaBest mu:Real:0-0.1 beta:Real:0.3-1 --maximize-metric --n-iters 20
