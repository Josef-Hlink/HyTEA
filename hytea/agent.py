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
        model: Model, optimizer: Optimizer, scheduler: StepLR, gamma: float, n_steps: int, device: torch.device
    ) -> None:
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.gamma = gamma
        self.n_steps = n_steps

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
        self.discounts = torch.tensor([self.gamma**i for i in range(self.n_steps+1)]).to(self.device)
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
            trajectory.add(dist.log_prob(action), value, reward, dist.entropy(), done)
            state = next_state
        return trajectory

    def _learn(self, trajectory: Trajectory, baseline=True) -> None:
        """ Learns from a trajectory using the actor-critic algorithm.

        ### Args:
        `Trajectory` trajectory: trajectory to learn from.
        """

        P, R, V, E, D = trajectory.unpack()
        
        G = torch.zeros(len(R), device=self.device)
        _r = 0
        for r in R.flip(0):
            _r = r + self.gamma * _r
            G = torch.cat((_r.unsqueeze(0), G[:-1]))

        # normalize rewards
        G = (G - G.mean()) / (G.std() + 1e-8)
        # baseline subtraction
        if baseline:
            G = G - V

        # entropy regularization
        P += 0.001 * E

        aGrad = P * G
        # critic loss
        cGrad = F.smooth_l1_loss(V, G.detach())

        # backward pass
        self.optimizer.zero_grad()
        (aGrad.sum() + cGrad).backward()
        self.optimizer.step()
        self.scheduler.step()

        return

    def _discount(self, tensor: torch.Tensor) -> float:
        """ Discounts the rewards. """
        assert tensor.dim() == 1, f'tensor must be 1-dimensional, got {tensor.dim()}'
        if tensor.shape[0] == self.n_steps+1:
            return (self.discounts * tensor).sum()
        return (self.discounts[:tensor.shape[0]] * tensor).sum().item()
