from time import perf_counter


from hytea.agent import DQNAgent
from hytea.model import Model
from hytea.environment import Environment
import torch


def main():
    print('Hello, world!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = Environment('LunarLander-v2', device=device)

    model = Model(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=64, num_layers=4, dropout_rate=0.1, hidden_activation='relu', output_activation='tanh')


    for rep in range(3):
        agent = DQNAgent(model=model, lr=1e-3, lr_decay=0.9, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.0001, optimizer='adam', device=device)
        print('-' * 80)
        train_start = perf_counter()
        train_reward = agent.train(num_episodes=1000, env=env)
        print(f'Training time: {perf_counter() - train_start:.2f} seconds')
        print(f'Average reward: {train_reward:.2f}')
        test_start = perf_counter()
        test_reward = agent.test(num_episodes=10)
        print(f'Evaluation time: {perf_counter() - test_start:.2f} seconds')
        print(f'Average reward: {test_reward:.2f}')
