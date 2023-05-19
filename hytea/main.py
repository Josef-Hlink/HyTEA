from time import perf_counter


from hytea.agent import ActorCriticAgent
from hytea.model import Model, ActorCriticModel
from hytea.environment import Environment
import torch


def main(caller: str = 'cli'):
    print('Hello, world!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    env = Environment('LunarLander-v2', device=device)

    model = ActorCriticModel(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=64, num_layers=2, dropout_rate=0, hidden_activation='relu', output_activation='softmax')

    histories = []

    for rep in range(3):
        agent = ActorCriticAgent(model=model, lr=1e-3, lr_decay=0.99, gamma=0.9, optimizer='adam', device=device)
        print('-' * 80)
        train_start = perf_counter()
        train_reward, history = agent.train(num_episodes=5000, env=env)
        histories.append(history)
        print(f'Training time: {perf_counter() - train_start:.2f} seconds')
        print(f'Average reward: {train_reward:.2f}')
        test_start = perf_counter()
        test_reward = agent.test(num_episodes=100)
        print(f'Evaluation time: {perf_counter() - test_start:.2f} seconds')
        print(f'Average reward: {test_reward:.2f}')

    if caller == 'jupyter':
        return histories

if __name__ == '__main__':
    main()
