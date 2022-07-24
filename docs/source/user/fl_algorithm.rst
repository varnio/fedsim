.. _custom_flalg:

Custom CentralFLAlgorithm
=========================

Any custome DataManager class should inherit from ``fedsim.fl.fl_algorithm.CentralFLAlgorithm`` (or its children) and implement its abstract methods.


A Simple Custom CentralFLAlgorithm
----------------------------------

.. code-block:: python

   from typing import Optional, Hashable, Mapping, Dict, Any
   from fedsim.fl.fl_algorithm import CentralFLAlgorithm

   class CustomFLAlgorithm(CentralFLAlgorithm):
       def __init__(
           self, data_manager, num_clients, sample_scheme, sample_rate, model_class, epochs, loss_fn,
           batch_size, test_batch_size, local_weight_decay, slr, clr, clr_decay, clr_decay_type,
           min_clr, clr_step_size, metric_logger, device, log_freq, other_arg, ... , *args, **kwargs,
       ):
           self.other_arg = other_arg

           super(FedAvg, self).__init__(
               data_manager, num_clients, sample_scheme, sample_rate, model_class, epochs, loss_fn,
               batch_size, test_batch_size, local_weight_decay, slr, clr, clr_decay, clr_decay_type,
               min_clr, clr_step_size, metric_logger, device, log_freq,
           )
           # make mode and optimizer
           model = self.get_model_class()().to(self.device)
           params = deepcopy(
               parameters_to_vector(model.parameters()).clone().detach())
           optimizer = SGD(params=[params], lr=slr)
           # write model and optimizer to server
           self.write_server('model', model)
           self.write_server('cloud_params', params)
           self.write_server('optimizer', optimizer)
           ...

       def send_to_client(self, client_id: int) -> Mapping[Hashable, Any]:
           """ returns context to send to the client corresponding to client_id.
               Do not send shared objects like server model if you made any
               before you deepcopy it.

           Args:
               client_id (int): id of the receiving client

           Raises:
               NotImplementedError: abstract class to be implemented by child

           Returns:
               Mapping[Hashable, Any]: the context to be sent in form of a Mapping
           """
           raise NotImplementedError

       def send_to_server(
           self, client_id: int, datasets: Dict[str, Iterable], epochs: int, loss_fn: nn.Module,
           batch_size: int, lr: float, weight_decay: float = 0, device: Union[int, str] = 'cuda',
           ctx: Optional[Dict[Hashable, Any]] = None, *args, **kwargs
       ) -> Mapping[str, Any]:
           """ client operation on the recieved information.

           Args:
               client_id (int): id of the client
               datasets (Dict[str, Iterable]): this comes from Data Manager
               epochs (int): number of epochs to train
               loss_fn (nn.Module): either 'ce' (for cross-entropy) or 'mse'
               batch_size (int): training batch_size
               lr (float): client learning rate
               weight_decay (float, optional): weight decay for SGD. Defaults to 0.
               device (Union[int, str], optional): Defaults to 'cuda'.
               ctx (Optional[Dict[Hashable, Any]], optional): context reveived from server. Defaults to None.

           Raises:
               NotImplementedError: abstract class to be implemented by child

           Returns:
               Mapping[str, Any]: client context to be sent to the server
           """
           raise NotImplementedError

       def receive_from_client(self, client_id: int, client_msg: Mapping[Hashable, Any], aggregator: Any):
           """ receive and aggregate info from selected clients

           Args:
               client_id (int): id of the sender (client)
               client_msg (Mapping[Hashable, Any]): client context that is sent
               aggregator (Any): aggregator instance to collect info

           Raises:
               NotImplementedError: abstract class to be implemented by child
           """
           raise NotImplementedError

       def optimize(self, aggregator: Any) -> Mapping[Hashable, Any]:
           """ optimize server mdoel(s) and return metrics to be reported

           Args:
               aggregator (Any): Aggregator instance

           Raises:
               NotImplementedError: abstract class to be implemented by child

           Returns:
               Mapping[Hashable, Any]: context to be reported
           """
           raise NotImplementedError

       def deploy(self) -> Optional[Mapping[Hashable, Any]]:
           """ return Mapping of name -> parameters_set to test the model

           Raises:
               NotImplementedError: abstract class to be implemented by child
           """
           raise NotImplementedError

       def report(
           self, dataloaders, metric_logger: Any, device: str, optimize_reports: Mapping[Hashable, Any],
           deployment_points: Optional[Mapping[Hashable, torch.Tensor]] = None
       ) -> None:
           """test on global data and report info

           Args:
               dataloaders (Any): dict of data loaders to test the global model(s)
               metric_logger (Any): the logging object (e.g., SummaryWriter)
               device (str): 'cuda', 'cpu' or gpu number
               optimize_reports (Mapping[Hashable, Any]): dict returned by optimzier
               deployment_points (Mapping[Hashable, torch.Tensor], optional): output of deploy method

           Raises:
               NotImplementedError: abstract class to be implemented by child
           """
           raise NotImplementedError


Integration with included cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically include your custom algorithm by the provided cli tool, you can place your class in a file under fedsim/fl/algorithms. Then, call it using option --algorithm. To deliver arguments to the **init** method of your custom algorithm, you can pass options in form of `--a-<arg-name>` where `<arg-name>` is the argument. Example

.. code-block:: bash

   fedsim fed-learn --algorithm CustomFLAlgorithm --a-other_arg <other_arg_value> ...
