"""pytorchexample: A Flower / PyTorch app."""

import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Print a compact experiment summary before training begins
    print("\n" + "=" * 72, flush=True)
    print("50-NODE HOMOGENEOUS FEDAVG BASELINE", flush=True)
    print("=" * 72, flush=True)
    print(f"Rounds              : {num_rounds}", flush=True)
    print(f"Training fraction   : {fraction_train:.2f}", flush=True)
    print(f"Evaluation fraction : {fraction_evaluate:.2f}", flush=True)
    print(f"Local epochs        : {context.run_config['local-epochs']}", flush=True)
    print(f"Batch size          : {context.run_config['batch-size']}", flush=True)
    print(f"Learning rate       : {lr}", flush=True)
    print("Partitioning         : IID", flush=True)
    print("Artificial delay    : none", flush=True)
    print("Client dropout      : none", flush=True)
    print("Aggregation          : synchronous FedAvg", flush=True)
    print("=" * 72 + "\n", flush=True)

    # Load global model
    global_model = Net()
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"[SERVER] Model parameters: {total_params:,}", flush=True)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    experiment_start = time.perf_counter()
    print("[SERVER] Starting federated training...", flush=True)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    total_duration = time.perf_counter() - experiment_start
    print("\n" + "=" * 72, flush=True)
    print("[SERVER] BASELINE RUN COMPLETE", flush=True)
    print(f"[SERVER] Total wall-clock time: {total_duration:.2f}s", flush=True)
    print("=" * 72 + "\n", flush=True)

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Print centralized global-model performance after each round
    print(
        f"[GLOBAL EVAL] round={server_round:02d} | "
        f"loss={test_loss:.4f} | accuracy={test_acc:.4f}",
        flush=True,
    )

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
