#!/usr/bin/env python3

from hytea.transitions import Transition, Trajectory

import torch
import torch.nn.functional as F
from gym import Env


"""
model that is very flexible and can take a lot of args
"""

from hytea.model import Model


class DQNAgent:

    def __init__(self,
        model: Model, lr: float, lr_decay: float, gamma: float, epsilon: float, epsilon_decay: float, epsilon_min: float,
        optimizer: str, use_GPU: bool
    ) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_GPU else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[optimizer](self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)
        
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        return
    
    def train(self, num_episodes: int, env: Env) -> None:
        self.env = env

        pass

    def sample_episode(self) -> Trajectory:
        trajectory = Trajectory()
        state, _ = self.env.reset()  # WATCH OUT: the reset method returns different stuff for different environments
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, trunc, _ = self.env.step(action)
            trajectory.add(Transition(state, action, reward, next_state, done))
            state = next_state



    def choose_action(self, state: torch.Tensor) -> int:
        return self.env.action_space.sample() if torch.rand(1) < self.epsilon else torch.argmax(self.model(state)).item()











    def anneal_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return
    