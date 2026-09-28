"""pytorchexample: A Flower / PyTorch app."""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    start_time = time.perf_counter()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = msg.content["config"]["lr"]

    print(
        f"[CLIENT {partition_id:02d}] TRAIN START | "
        f"partition={partition_id}/{num_partitions - 1} | "
        f"epochs={local_epochs} | batch_size={batch_size} | lr={learning_rate}",
        flush=True,
    )

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        learning_rate,
        device,
    )

    duration = time.perf_counter() - start_time
    num_examples = len(trainloader.dataset)

    print(
        f"[CLIENT {partition_id:02d}] TRAIN END   | "
        f"samples={num_examples} | loss={train_loss:.4f} | "
        f"duration={duration:.2f}s",
        flush=True,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": num_examples,
        "train_duration_sec": duration,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    start_time = time.perf_counter()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    print(
        f"[CLIENT {partition_id:02d}] EVAL START  | "
        f"partition={partition_id}/{num_partitions - 1} | "
        f"batch_size={batch_size}",
        flush=True,
    )

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    duration = time.perf_counter() - start_time
    num_examples = len(valloader.dataset)

    print(
        f"[CLIENT {partition_id:02d}] EVAL END    | "
        f"samples={num_examples} | loss={eval_loss:.4f} | "
        f"accuracy={eval_acc:.4f} | duration={duration:.2f}s",
        flush=True,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": num_examples,
        "eval_duration_sec": duration,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
