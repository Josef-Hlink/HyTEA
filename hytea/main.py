

from hytea.agent import DQNAgent
from hytea.model import Model
from hytea.environments import BipedalWalkerEnv
import torch


def main():
    print('Hello, world!')    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = BipedalWalkerEnv(device=device)

    model = Model(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=64, num_layers=2, dropout_rate=0.1, hidden_activation='relu', output_activation='tanh')

    agent = DQNAgent(model=model, lr=1e-3, lr_decay=0.9, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.0001, optimizer='adam', device=device)

    agent.train(num_episodes=1000, env=env)
