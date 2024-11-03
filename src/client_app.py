import warnings
from typing import List, Tuple

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.client import Client
from flwr.common import Context
from opacus import PrivacyEngine

from src.task import (
    CustomResNet,
    encrypt_and_serialize_parameters,
    get_data_size_in_mb,
    get_dataset_name,
    load_data,
    set_weights,
    test,
    train,
)

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
        in_channels,
        num_classes,
    ) -> None:
        super().__init__()
        self.model = CustomResNet(in_channels=in_channels, num_classes=num_classes)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, dict]:
        model = self.model
        set_weights(model, parameters)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        privacy_engine = PrivacyEngine(secure_mode=False)
        (model, optimizer, self.train_loader,) = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        epsilon = train(
            model,
            self.train_loader,
            privacy_engine,
            optimizer,
            self.target_delta,
            device=self.device,
        )

        if epsilon is not None:
            print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
        else:
            print("Epsilon value not available.")

        # Encrypt and serialize model parameters
        updated_parameters = [param.data.cpu() for param in model.parameters()]
        encrypted_and_serialized_parameters = encrypt_and_serialize_parameters(
            updated_parameters
        )

        # Return encrypted parameters, dataset size, and empty metrics
        return (
            encrypted_and_serialized_parameters,
            len(self.train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    task, train_loader, test_loader, labels, in_channels = load_data(
        partition_id, context.node_config["num-partitions"], get_dataset_name()
    )
    return FlowerClient(
        train_loader,
        test_loader,
        context.run_config["target-delta"],
        noise_multiplier,
        context.run_config["max-grad-norm"],
        in_channels=in_channels,
        num_classes=len(labels),
    ).to_client()


app = ClientApp(client_fn=client_fn)
