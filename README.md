# pytorch-research-template

This template structure is opinionated and is designed to be used for research projects. It consists of everything I use and recommend using for modern 2022 research projects with PyTorch.

## Opinionated features

- [x] PyTorch Lightning
- [x] Hydra
- [x] SLURM
- [x] Pre-commit


## Setup

Install miniconda and create a new environment for the project:

```bash
conda create -n <env_name> python=3.8
conda activate <env_name>
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Install the pre-commit hooks:

```bash
pre-commit install
```

## Training

### Local training

To train a model, run the following command:

```bash
python main.py
```

### Distributed training

To train a model on multiple GPUs, run the following command:

```bash
sbatch scripts/train.sh
```
