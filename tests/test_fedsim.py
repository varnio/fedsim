import math

from torch.utils.tensorboard import SummaryWriter

from fedsim.distributed.centralized.training import FedAvg
from fedsim.distributed.data_management import BasicDataManager
from fedsim.models.mcmahan_nets import cnn_cifar100
from fedsim.scores import accuracy
from fedsim.scores import cross_entropy


def test_main():
    n_clients = 1000

    dm = BasicDataManager("./data", "cifar100", n_clients)
    sw = SummaryWriter()

    alg = FedAvg(
        data_manager=dm,
        num_clients=n_clients,
        sample_scheme="uniform",
        sample_rate=0.01,
        model_class=cnn_cifar100,
        epochs=1,
        loss_fn=cross_entropy,
        batch_size=32,
        metric_logger=sw,
        device="cpu",
    )

    alg.hook_global_score_function("test", "accuracy", accuracy)
    for key in dm.get_local_splits_names():
        alg.hook_local_score_function(key, "accuracy", accuracy)

    for key, value in alg.train(rounds=1).items():
        if "accuracy" in key:
            assert value >= 0
        elif "loss" in key:
            assert 0 <= value < 2 * math.log(100)
