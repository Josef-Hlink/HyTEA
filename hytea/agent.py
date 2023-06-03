from hytea.transitions import Trajectory
from hytea.model import Model
from hytea.environment import Environment

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical


class Agent:
    """ Agent that learns to play an environment using the actor-critic algorithm. """

    def __init__(self,
        model: Model, optimizer: Optimizer, scheduler: StepLR,
        gamma: float, ent_reg_weight: float, bl_sub: bool,
        device: torch.device
    ) -> None:
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.gamma = gamma
        self.ent_reg_weight = ent_reg_weight
        self.bl_sub = bl_sub

        self.trained = False
        self.env = None
        return
    
    def train(self, num_episodes: int, env: Environment) -> list[float]:
        """ Train the agent for a number of episodes. 
        
        ### Args:
        `int` num_episodes: number of episodes to train for.
        `Environment` env: environment to train on.

        ### Returns:
        `list[float]`: history of total reward per episode.
        """
        self.env = env
        history = []

        for _ in range(num_episodes):
            trajectory = self._sample_episode()
            history.append(trajectory.total_reward)
            self._learn(trajectory)

        self.trained = True
        return history
    
    def test(self, num_episodes: int) -> float:
        """ Test the agent for a number of episodes.

        ### Args:
        `int` num_episodes: number of episodes to test for.

        ### Returns:
        `float`: average reward per episode.
        """
        assert self.trained, 'Agent must be trained before testing.'
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
        trajectory = Trajectory(self.env.spec.max_episode_steps, self.device)
        state, done = self.env.reset(), False
        while not done:
            probs, value = self.model(state)
            dist = Categorical(probs)
            action = dist.sample()
            next_state, reward, done, trunc, _ = self.env.step(action.item())
            done = done or trunc
            trajectory.add(dist.log_prob(action), dist.entropy(), value, reward)
            state = next_state
        return trajectory

    def _learn(self, trajectory: Trajectory, baseline=True) -> None:
        """ Learns from a trajectory using the actor-critic algorithm.

        ### Args:
        `Trajectory` trajectory: trajectory to learn from.
        """

        P, E, V, R = trajectory.unpack()         # get trajectory tensors
        
        G, g = [], 0                             # compute returns with clever math magic
        for r in R.flip(0):
            g = r + self.gamma * g
            G.insert(0, g)
        G = torch.tensor(G, device=self.device)  # cast to tensor
        G = (G - G.mean()) / (G.std() + 1e-8)    # normalize

        if self.bl_sub: G -= V                   # subtract baseline
        P += self.ent_reg_weight * E             # add entropy regularization

        al = -P * G                              # actor loss
        cl = F.smooth_l1_loss(V, G)              # critic loss

        self.optimizer.zero_grad()
        (al + cl).sum().backward()               # backprop
        self.optimizer.step()
        self.scheduler.step()

        return
