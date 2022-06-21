from torch.utils.tensorboard import SummaryWriter

from fedsim.distributed.centralized.training.fedavg import FedAvg
from fedsim.distributed.data_management.basic_data_manager import BasicDataManager
from fedsim.models.mcmahan_nets import cnn_cifar100


def test_main():
    n_clients = 100

    dm = BasicDataManager("./data", "cifar100", n_clients)
    sw = SummaryWriter()

    alg = FedAvg(
        data_manager=dm,
        num_clients=n_clients,
        sample_scheme="uniform",
        sample_rate=0.01,
        model_class=cnn_cifar100,
        epochs=5,
        loss_fn="ce",
        batch_size=32,
        metric_logger=sw,
        device="cpu",
    )
    assert alg.train(rounds=1) == 0
