0.3.1 (2022-08-09)
------------------

* added advanced learning rate schedulers
* properly tested r2r lr scheduler

0.3.0 (2022-08-09)
------------------

* added fine-tunning to cli, `fed-tune`
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
