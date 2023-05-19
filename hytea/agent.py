#!/usr/bin/env python3

from hytea.transitions import Trajectory
from hytea.model import ActorCriticModel
from hytea.environment import Environment

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.distributions import Categorical
import numpy as np
import tqdm

class ActorCriticAgent:

    def __init__(self,
        model: ActorCriticModel, lr: float, lr_decay: float, gamma: float, optimizer: str, device: torch.device
    ) -> None:
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer: Optimizer = dict(
            adam = torch.optim.Adam,
            sgd = torch.optim.SGD
        )[optimizer](self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)
        
        self.gamma = gamma
        self.nSteps = 100  # bootstrap steps

        return
    
    def train(self, num_episodes: int, env: Environment) -> float:
        """ Train the agent for a number of episodes. 
        
        ### Args:
        `int` num_episodes: number of episodes to train for.
        `Environment` env: environment to train on.

        ### Returns:
        `float`: average reward per episode.
        """
        self.env = env
        self.discounts = torch.tensor([self.gamma**i for i in range(self.env.spec.max_episode_steps)]).to(self.device)
        total_reward = 0
        history = []

        for _ in tqdm.tqdm(range(num_episodes)):
            trajectory = self._sample_episode()
            total_reward += trajectory.total_reward
            history.append(trajectory.total_reward)
            self._learn(trajectory)
        return total_reward / num_episodes, history
    
    def test(self, num_episodes: int) -> float:
        """ Test the agent for a number of episodes.

        ### Args:
        `int` num_episodes: number of episodes to test for.

        ### Returns:
        `float`: average reward per episode.
        """
        assert self.env is not None, 'Agent has not been trained yet.'
        total_reward = 0
        for _ in range(num_episodes):
            trajectory = self._sample_episode()
            total_reward += trajectory.total_reward
        return total_reward / num_episodes

    def _sample_episode(self) -> Trajectory:
        """ Samples an episode from the environment.
        
        ### Returns:
        `Trajectory`: the sampled episode.
        """
        trajectory = Trajectory(self.env.observation_space.shape, self.env.spec.max_episode_steps, self.device)
        state, done = self.env.reset(), False
        while not done:
            action = self._choose_action(state)
            next_state, reward, done, trunc, _ = self.env.step(action)
            done = done or trunc
            trajectory.add(state, action, reward, next_state, done)
            state = next_state
        return trajectory

    def _choose_action(self, state: torch.Tensor) -> int:
        """ Chooses an action based on the current state. """
        with torch.no_grad(): probs, _ = self.model(state)
        return Categorical(probs).sample().item()

    def _learn(self, trajectory: Trajectory) -> None:
        """ Learns from a trajectory using the actor-critic algorithm.

        ### Args:
        `Trajectory` trajectory: trajectory to learn from.
        """

        S, A, R, S_, D = trajectory.unpack()
        
        # forward pass to get action probabilities and state values
        P, V = self.model(S)
        V = V.squeeze()

        # get distributions from probabilities
        dist = Categorical(P)

        with torch.no_grad():
            V_ = torch.roll(V, -1)
            V_[-1] = 0
        
        G = torch.zeros(len(R), device=self.device)
        for i in range(len(R)):
            # slice out the next nSteps transitions
            slc: slice = slice(i, i + self.nSteps)
            # substitute value of last state with bootstrap value
            _R = torch.cat((R[slc], V_[slc][-1] * (~D[slc][-1]).unsqueeze(-1)))
            # sum of discounted rewards
            # print(_R)
            G[i] = sum(self._discount(_R))

        # actor gradient
        aGrad = -dist.log_prob(A) * (G - V)
        # add entropy regularization
        aGrad -= 0.1 * dist.entropy()
        # critic gradient
        cGrad = F.mse_loss(V, G)

        # backward pass
        self.optimizer.zero_grad()
        (aGrad + cGrad).mean().backward()
        self.optimizer.step()
        self.scheduler.step()

        return

    def _discount(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Discounts the Rewards. """
        assert tensor.dim() == 1, f'tensor must be 1-dimensional, got {tensor.dim()}'
        return self.discounts[:tensor.shape[0]] * tensor
