0.6.1 (2022-08-17)
------------------

* fixed bug in ``partition_global_data`` of ``BasicDataManager``
* some changes in default values for better log storage and aggregation

0.6.0 (2022-08-16)
------------------

* changed the name of cli directory
* added cli tests
* added support for pytorch original lr schedulers
* improved docs
* added version option to fedsim-cli

0.5.0 (2022-08-15)
------------------

* completed lr schedulers and generalized them for all levels
* changed some argument names and default values

0.4.1 (2022-08-12)
------------------

* fixed bugs with mismatched loss_fn argument name in cli commands
* changed all ``eval_freq`` arguemnts to unified ``log_req``

0.4.0 (2022-08-12)
------------------

* changed the structure of scores and losses
* made it possible to hook multiple local and global scores

0.3.1 (2022-08-09)
------------------

* added advanced learning rate schedulers
* properly tested r2r lr scheduler

0.3.0 (2022-08-09)
------------------

* added fine-tuning to cli, `fed-tune`
* cleaner cli
* made optimizers and schedulers user definable
* improved logging


0.2.0 (2022-08-01)
------------------

* cleaned the API reference in docs
* changed cli name to `fedsim-cli`
* improved documentation
* improved importing
* changed the way custom objects are passed to cli

0.1.4 (2022-07-23)
------------------

* changed FLAlgorithm to CentralFLAlgorithm for more clearity
* set default device to cuda if available otherwise to cpu in fed-learn cli
* fix wrong superclass names in demo
* fix the confusion with `save_dir` and `save_path` in DataManager classes


0.1.3 (2022-07-08)
------------------

* the documentation is redesigned and mostly automated.
* documentation now is available at https://fesim.varnio.com
* added code of coduct from github tempalate


0.1.2 (2022-07-05)
------------------

* changed ownership of repo from fedsim-dev to varnio


0.1.1 (2022-06-22)
------------------

* added fedsim.scores which wraps torch loss functions and sklearn scores
* moved reporting mechanism of distributed algorithm for supporting auto monitor
* added AppendixAggregator which is used to hold metric scores and report final results
* apply a patch for wrong pypi supported python versions

0.1.0 (2022-06-21)
------------------

* First major pre-release.
* The package is restructured
* docs is updated and checked to pass through tox steps



0.0.4 (2022-06-14)
------------------

* Fourth release on PyPI.
