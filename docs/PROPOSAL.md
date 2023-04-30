# Proposal (HyTEA)

## Team

- Thomas Blom
- Josef Hamelink
- Shreyansh Sharma

## Idea

Hyperparameter tuning is a very important part of machine learning, but it is also very time consuming.
We want to use genetic algorithms to optimize the hyperparameters of a neural network, such as number of layers, number of neurons per layer, activation functions, learning rate, etc.
The neural network will be used by Deep Q-Learning agents to play a game.

We do not yet know what game we want to use as an environment for the agents, but it should be a simple game that that can be completed within a small amount of time, as we want the focus to be on iterating over the hyperparameters and not on the game itself.

Some ideas for games are:

- Snake
- Tetris
- Tic-Tac-Toe

## Research

We want to spend time researching the latest developments in Evolutionary Computation and Hyperparameter Tuning, and try to implement some of the ideas in our project.

## Possible extensions

If all goes well, we might look into encoding additional hyperparameters specific to DQN agents into the candidate solution bitstrings, such as the discount factor, exploration strategy, etc.
