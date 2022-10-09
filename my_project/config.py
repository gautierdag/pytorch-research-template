from pathlib import Path

# we use pydantic dataclasses to parse Path objects correctly
from pydantic.dataclasses import dataclass


@dataclass
class Paths:
    input_dir: Path
    output_dir: Path


@dataclass
class Model:
    learning_rate: float


@dataclass
class Train:
    batch_size: int
    max_epochs: int


@dataclass
class MyProjectConfig:
    run_name: str
    seed: int
    gpus: str  # use "1" to use only one gpu
    num_workers: int  # number of workers for dataloader

    paths: Paths  # input and output directories
    model: Model  # model parameters constant for both pretrain and nlu
    train: Train  # Settings specific to pretraining

    fast: bool = False  # if True, disable wandb logging
