import gymnasium as gym
import torch


class Environment:

    def __init__(self, gym_env: str, device: torch.device):
        self.gym_env: gym.Env = gym.make(gym_env)
        self.device = device
    
    def reset(self) -> torch.Tensor:
        state, _ = self.gym_env.reset()
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def step(self, action: int) -> torch.Tensor:
        state, reward, done, trunc, info = self.gym_env.step(action)
        return torch.tensor(state, dtype=torch.float32, device=self.device), reward, done, trunc, info
    
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.gym_env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Space:
        return self.gym_env.action_space
    
    @property
    def spec(self) -> gym.envs.registration.EnvSpec:
        return self.gym_env.spec