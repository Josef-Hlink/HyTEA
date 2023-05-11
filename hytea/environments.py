import gym.envs.box2d as environments
import torch

# aliases
BipedalWalker = environments.BipedalWalker
CarRacing = environments.CarRacing
LunarLander = environments.LunarLander

class OurOwnEnv():
    
    def __init__(self, device: torch.device):
        self.device = device
        return
    
    def step(self, action):
        state, reward, done, info = super().step(action)
        return torch.Tensor(state, self.device), reward, done, False, info

    def reset(self):
        state = super().reset()
        return state, None


class BipedalWalkerEnv(BipedalWalker, OurOwnEnv):

    def __init__(self, device: torch.device):
        super().__init__()
        OurOwnEnv.__init__(self, device)
        return
    
    def step(self, action: int) -> torch.Tensor:
        state, reward, done, trunc, info = super().step(action)
        return OurOwnEnv.step(self, action)
