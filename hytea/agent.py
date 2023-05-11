#!/usr/bin/env python3

from hytea.transitions import Trajectory
from hytea.model import Model

import torch
from gym import Env
from torch.nn import functional as F
from tqdm.auto import tqdm

class DQNAgent:

    def __init__(self,
        model: Model, lr: float, lr_decay: float, gamma: float, epsilon: float, epsilon_decay: float, epsilon_min: float,
        optimizer: str, device: torch.device
    ) -> None:
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[optimizer](self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)
        
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # tqdm progress bar
        self.pbar = tqdm()
        self.pbar.set_description('Training')
        return
    
    def train(self, num_episodes: int, env: Env) -> None:
        self.env = env
        # use tqdm progress bar
        for _ in self.pbar(range(num_episodes)):
            trajectory = self._sample_episode()
            pbar.desc = f'Training (episode: {trajectory.l}, reward: {trajectory.R.sum().item()})'
            self._learn(trajectory)
            self._anneal_epsilon()
            self.scheduler.step()
        return

    def _sample_episode(self) -> Trajectory:
        trajectory = Trajectory(self.env.observation_space.shape, self.env.spec.max_episode_steps)
        state, done = self.env.reset(), False
        while not (done or trunc):
            action = self._choose_action(state)
            next_state, reward, done, trunc, _ = self.env.step(action)
            trajectory.add(state, action, reward, next_state, done)
            state = next_state
        return trajectory

    def _choose_action(self, state: torch.Tensor) -> int:
        return self.env.action_space.sample() if torch.rand(1) < self.epsilon else torch.argmax(self.model(state)).item()

    def _anneal_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return
    
    def _learn(self, trajectory: Trajectory) -> None:
        states, actions, rewards, next_states, dones = trajectory.get_tensors()
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1).detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return
    