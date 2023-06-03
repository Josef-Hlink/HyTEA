# HyTEA

Hyperparameter Tuning using Evolutionary Algorithms

Contributors:

- Thomas Blom
- Josef Hamelink
- Levi Peeters
- Shreyansh Sharma

## Description

HyTEA is a hyperparameter tuning library that uses evolutionary algorithms to find the optimal hyperparameters (including architecture) for a given game.

## Setup

### Requirements

You'll need Python 3.11.0 or higher.

### Installation

1. Clone the repository
2. Optionally create & activate a virtual environment
3. Install the requirements using `pip install -e .`

## Usage

After installation: the following scripts will become available in your virtual environment:

```bash
hytea run  # runs the complete hyperparameter tuning process
hytea test  # tests the default hyperparameters (single run)
hytea decode  # decodes a given hyperparameter string
```
