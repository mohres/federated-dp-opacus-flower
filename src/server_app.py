import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from flwr.common import (
    Context,
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from src.task import (
    CustomResNet,
    deserialize_and_decrypt_parameters,
    get_centralized_eval_dataset,
    get_data_size_in_mb,
    get_dataset_name,
    get_weights,
    set_weights,
    test,
)

# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SMPCFedAvg(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average with decryption."""

        if not results:
            return None, {}

        # Check for failures
        if not self.accept_failures and failures:
            return None, {}

        # Decrypt parameters from each client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        weights_results = [
            (deserialize_and_decrypt_parameters(parameters), num_examples)
            for parameters, num_examples in weights_results
        ]

        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate metrics if provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    task, test_loader = get_centralized_eval_dataset(get_dataset_name())

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_weights(model, parameters)  # Update model with the latest parameters

        loss, accuracy = test(
            model, test_loader=test_loader, device=DEVICE, task=task, desc="Evaluating"
        )
        return loss, {"accuracy": accuracy}

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    model = CustomResNet(3, 8)
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = SMPCFedAvg(
        fraction_fit=fraction_fit,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(model),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
