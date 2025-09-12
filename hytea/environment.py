import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import torch


class Environment:
    def __init__(self, env_name: str, device: torch.device):
        assert env_name in [
            'CartPole-v1',
            'Acrobot-v1',
            'LunarLander-v3',
        ], 'Environment not supported. See README.md for supported environments.'

        self.gym_env: gym.Env = gym.make(env_name)
        self.device = device

    def reset(self) -> torch.Tensor:
        state, _ = self.gym_env.reset()
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]:
        state, reward, done, trunc, info = self.gym_env.step(action)
        return (
            torch.tensor(state, dtype=torch.float32, device=self.device),
            float(reward),
            done,
            trunc,
            info,
        )

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.gym_env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.gym_env.action_space

    @property
    def spec(self) -> EnvSpec:
        assert self.gym_env.spec is not None
        return self.gym_env.spec
