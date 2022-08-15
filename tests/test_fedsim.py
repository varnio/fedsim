import math
from functools import partial

from logall import TensorboardLogger

from fedsim.distributed.centralized import AdaBest
from fedsim.distributed.centralized import FedAvg
from fedsim.distributed.centralized import FedDyn
from fedsim.distributed.centralized import FedNova
from fedsim.distributed.centralized import FedProx
from fedsim.distributed.data_management import BasicDataManager
from fedsim.losses import CrossEntropyLoss
from fedsim.models.mcmahan_nets import cnn_cifar100
from fedsim.scores import Accuracy


def alg_hook(alg, dm):
    for key in dm.get_local_splits_names():
        alg.hook_local_score(
            partial(Accuracy, log_freq=100),
            split_name=key,
            score_name="accuracy",
        )
    for key in dm.get_global_splits_names():
        alg.hook_global_score(
            partial(Accuracy, log_freq=100),
            split_name=key,
            score_name="accuracy",
        )


def acc_check(alg):
    for key, value in alg.train(rounds=1).items():
        if "accuracy" in key:
            assert value >= 0
        elif "loss" in key:
            assert 0 <= value < 2 * math.log(100)


def test_algs():
    n_clients = 5000
    dm = BasicDataManager("./data", "cifar100", n_clients)
    sw = TensorboardLogger(path=None)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    common_cfg = dict(
        data_manager=dm,
        num_clients=2,
        sample_scheme="uniform",
        sample_rate=1.0,
        model_def=cnn_cifar100,
        epochs=1,
        criterion_def=partial(CrossEntropyLoss, log_freq=100),
        batch_size=32,
        metric_logger=sw,
        device=device,
    )
    # test fedavg
    alg = FedAvg(**common_cfg)
    alg_hook(alg, dm)
    acc_check(alg)
    del alg

    # test fedprox
    alg = FedProx(**common_cfg)
    alg_hook(alg, dm)
    acc_check(alg)
    del alg

    # test fednova
    alg = FedNova(**common_cfg)
    alg_hook(alg, dm)
    acc_check(alg)
    del alg

    # test feddyn
    alg = FedDyn(**common_cfg)
    alg_hook(alg, dm)
    acc_check(alg)
    del alg

    # test AdaBest
    alg = AdaBest(**common_cfg)
    alg_hook(alg, dm)
    acc_check(alg)
    del alg
