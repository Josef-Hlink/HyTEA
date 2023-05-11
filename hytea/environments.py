from gym.envs.box2d import BipedalWalker, CarRacing, LunarLander
import torch


class BipedalWalkerEnv(BipedalWalker):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        return
    
    def reset(self) -> torch.Tensor:
        state, _ = super().reset()
        return torch.tensor(state, torch.float32, self.device)

    def step(self, action: int) -> torch.Tensor:
        state, reward, done, trunc, info = super().step(action)
        return torch.tensor(state, torch.float32, self.device), reward, done, trunc, info

class CarRacingEnv(CarRacing):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        return
    
    def reset(self) -> torch.Tensor:
        state, _ = super().reset()
        return torch.tensor(state, torch.float32, self.device)

    def step(self, action: int) -> torch.Tensor:
        state, reward, done, trunc, info = super().step(action)
        return torch.tensor(state, torch.float32, self.device), reward, done, trunc, info
    
class LunarLanderEnv(LunarLander):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        return
    
    def reset(self) -> torch.Tensor:
        state, _ = super().reset()
        return torch.tensor(state, torch.float32, self.device)

    def step(self, action: int) -> torch.Tensor:
        state, reward, done, trunc, info = super().step(action)
        return torch.tensor(state, torch.float32, self.device), reward, done, trunc, info
