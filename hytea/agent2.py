from hytea.utils import DotDict
from hytea.model import Model
from hytea.environment import Environment
from hytea.transitions2 import Trajectory

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
        self.entropy_weight = 0.001
        self.baseline = True
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
        if not self.trained:
            raise RuntimeError('Agent is not trained.')
        
        history = []
        
        for _ in range(num_episodes):
            trajectory = self._sample_episode()
            history.append(trajectory.total_reward)
            
        return sum(history)/num_episodes
    
    def _sample_episode(self) -> Trajectory:
        """ Sample an episode from the environment. """
        trajectory = Trajectory(self.env.spec.max_episode_steps, self.device)
        state, done = self.env.reset(), False
        while not done:
            probs, value = self.model(state)
            dist = Categorical(probs)
            action = dist.sample()
            state_, reward, done, trunc, *_ = self.env.step(action.item())
            done = done or trunc
            trajectory.add(dist.log_prob(action), value, reward, dist.entropy())
            state = state_
        return trajectory
    
    def _learn(self, trajectory: Trajectory) -> None:
        """ Learn from a trajectory of transitions.
        
        ### Args:
        `Trajectory` trajectory: trajectory to learn from.
        """
        
        P, V, R, E = trajectory.unpack()
        
        # Compute returns
        G = []
        _r = 0
        for r in R.flip(0):
            _r = r + self.gamma * _r
            # insert at the beginning of the tensor
            G.insert(0, _r)
        
        G = torch.tensor(G, dtype=torch.float32, device=self.device)
        
        # normalize returns
        G = (G - G.mean()) / (G.std() + 1e-10)
        
        loss = self.loss3(G, P, V, E)
        
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return
    
    def loss1(self, G: torch.Tensor, P: torch.Tensor, V: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # compute loss
        loss = 0
        for p, v, g, e in zip(P, V, G, E):
            # entropy regularization
            p = p + self.entropy_weight * e
            
            # baseline subtraction
            if self.baseline:
                g = g - v
            
            # actor loss
            actor_loss = -p * g
            
            # critic loss
            critic_loss = F.smooth_l1_loss(v, g)
            
            # total loss
            loss += actor_loss + critic_loss
        return loss
    
    def loss2(self, G: torch.Tensor, P: torch.Tensor, V: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        if self.baseline:
            G = G - V
            
        P = P + self.entropy_weight * E
        
        actor_loss = -P * G
        
        critic_loss = F.smooth_l1_loss(V, G)
        
        loss = (actor_loss.sum() + critic_loss.sum())
        
        return loss
    
    def loss3(self, G: torch.Tensor, P: torch.Tensor, V: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        if self.baseline:
            G = G - V
        
        P = P + self.entropy_weight * E
        
        actor_loss = -P * G
        
        critic_loss = F.smooth_l1_loss(V, G)
        
        loss = (actor_loss + critic_loss).sum()
        
        return loss

    
    