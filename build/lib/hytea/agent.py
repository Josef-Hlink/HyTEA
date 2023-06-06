#!/usr/bin/env python3

from hytea.transitions import Trajectory
from hytea.model import Model
from hytea.environment import Environment

import torch
from torch.nn import functional as F
from torch.optim import Optimizer


class DQNAgent:

    def __init__(self,
        model: Model, lr: float, lr_decay: float, gamma: float, epsilon: float, epsilon_decay: float, epsilon_min: float,
        optimizer: str, device: torch.device
    ) -> None:
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer: Optimizer = dict(
            adam = torch.optim.Adam,
            sgd = torch.optim.SGD
        )[optimizer](self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)
        
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
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
        total_reward = 0
        for _ in range(num_episodes):
            trajectory = self._sample_episode()
            total_reward += trajectory.total_reward
            self._learn(trajectory)
            self._anneal_epsilon()
        return total_reward / num_episodes
    
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
        return self.env.action_space.sample() if torch.rand(1) < self.epsilon else torch.argmax(self.model(state)).item()

    def _anneal_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return
    
    def _learn(self, trajectory: Trajectory) -> None:
        """ Learns from a trajectory.

        ### Args:
        `Trajectory` trajectory: trajectory to learn from.
        """
        S, A, R, S_, D = trajectory.unpack()
        Q = self.model(S).gather(1, A)
        Q_ = self.model(S_).max(1, keepdim=True).values
        loss = F.mse_loss(Q, R + self.gamma * Q_ * ~D)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return
    