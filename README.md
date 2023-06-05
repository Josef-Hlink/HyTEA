# HyTEA

Hyperparameter Tuning using Evolutionary Algorithms

Contributors:

- Thomas Blom
- Josef Hamelink
- Levi Peeters
- Shreyansh Sharma

## Description

HyTEA is a hyperparameter tuning library that uses evolutionary algorithms to find the optimal hyperparameters (including architecture) for a given game environment.
It is built from scratch with [torch](https://pytorch.org/) and tears.
For now, our CLI only supports [OpenAI Gymnasium](https://gymnasium.farama.org/) environments (see [list](#list-of-supported-environments)).
But it should be easy to extend to other environments.

## Setup

1. Clone the repository

2. Create & activate a virtual environment

You'll need Python 3.11.0 or higher.
The easiest way to get it (if you don't already have it) is to create a virtual environment using [conda](https://docs.conda.io/en/latest/):

```bash
conda create -n hytea python==3.11.0
conda activate hytea
```

If you already have Python 3.11.0, just do:
    
```bash
python -m venv hytea
source hytea/bin/activate
```

3. Install the requirements using:
```bash
pip install -e .
```

## Usage

After installation: the following scripts will become available in your virtual environment:

```bash
hytea run  # runs the complete hyperparameter tuning process
hytea test  # tests the default hyperparameters (single run)
hytea decode  # decodes a given hyperparameter string
```

## List of supported environments

- [x] CartPole-v1
- [x] Acrobot-v1
- [x] LunarLander-v2
- [ ] BipedalWalker-v3
