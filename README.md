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

Clone and navigate to this repository and simply run `uv sync` if you have [uv](https://uv.readthedocs.io/en/latest/) installed.
Alternatively, you can manually create a virtual environment and pull the dependencies from [pyproject.toml](pyproject.toml) like so:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> [!CAUTION]
> if you run into issues pulling the "gymnasium[box2d]" dependency,
> you may need to `sudo apt install swig` (or your system's equivalent).

## Usage

After installation, the following scripts will become available in your virtual environment:

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
