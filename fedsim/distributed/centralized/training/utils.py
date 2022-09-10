"""
Distributed Centralized Trainign Utils
--------------------------------------
"""


def serial_aggregation(
    server_storage,
    client_id,
    client_msg,
    train_split_name,
    aggregator,
    train_weight=None,
    other_weight=None,
):
    """To serially aggregate received message from a client

    Args:
        server_storage (Storage): server storage object
        client_id (int): client id.
        client_msg (Mapping): client message.
        train_split_name (str): name of the training split on clients
        aggregator (SerialAggregator): a serial aggregator to accumulate info.
        train_weight (float, optional): aggregation weight for trianing parameters.
            If not specified, uses sample number. Defaults to None.
        other_weight (float, optional): aggregation weight for any other
            factor/metric. If not specified, uses sample number. Defaults to None.

    Returns:
        bool: success of aggregation.

    """
    params = client_msg["local_params"].clone().detach().data
    diverged = client_msg["diverged"]
    metrics = client_msg["metrics"]
    n_samples = client_msg["num_samples"]

    if diverged:
        return False

    if train_weight is None:
        train_weight = n_samples[train_split_name]

    if train_weight > 0:
        aggregator.add("local_params", params, train_weight)
        for split_name, metrics in metrics.items():
            if other_weight is None:
                other_weight = n_samples[split_name]
            for key, metric in metrics.items():
                aggregator.add(f"clients.{split_name}.{key}", metric, other_weight)

    # purge client info
    del client_msg

    return True
