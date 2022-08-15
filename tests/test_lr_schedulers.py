from fedsim.lr_schedulers import ReduceLROnPlateau


def test_reduce_lr_on_plateau():
    init_lr = 0.1
    patience = 5
    lr_sch = ReduceLROnPlateau(init_lr, factor=0.1, patience=patience, verbose=True)
    for i in range(patience + 2):
        lr_sch.step(100 + i)
    assert lr_sch.get_the_last_lr()
