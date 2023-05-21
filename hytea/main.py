from time import perf_counter

from hytea import Environment, ActorCriticModel, ActorCriticAgent
from hytea.utils import DotDict
import torch


def main(config: DotDict, caller: str = 'cli') -> None | list:
    """ Main function for the hytea package. 
    
    ### Args:
    `DotDict` config: configuration dictionary.
    `str` caller: caller of the function. Either 'cli' or 'jupyter'.

    ### Returns:
    `None` if caller is 'cli', `list` of histories if caller is 'jupyter'.
    """
    if caller == 'cli':
        histories = []
    for _ in range(config.exp.num_runs):
        history = run(config, caller)
        if caller == 'cli':
            print(f'avg. reward: {sum(history)/len(history):.2f}')
            histories.append(history)
    return histories if caller == 'jupyter' else None


def run(config: DotDict, caller: str = 'cli') -> None | list:
    """ Run the agent for a number of episodes.

    ### Args:
    `DotDict` config: configuration dictionary.
    `str` caller: caller of the function. Either 'cli' or 'jupyter'.

    ### Returns:
    `None` if caller is 'cli', `list` of histories if caller is 'jupyter'.
    """
    
    device = torch.device('cpu')

    env = Environment('LunarLander-v2', device=device)

    model = ActorCriticModel(
        input_size = env.observation_space.shape[0],
        output_size = env.action_space.n,
        hidden_size = config.network.hidden_size,
        hidden_activation = config.network.hidden_activation,
        output_activation = config.network.output_activation,
        num_layers = config.network.num_layers,
        dropout_rate = config.network.dropout_rate,
    ).to(device)

    optimizer: torch.optim.Optimizer = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }[config.optimizer.type](model.parameters(), lr=config.optimizer.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.optimizer.lr_decay)

    agent = ActorCriticAgent(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        gamma = config.agent.gamma,
        n_steps = config.agent.n_steps,
        device = device
    )

    if caller == 'cli':
        start = perf_counter()
        history = agent.train(num_episodes=config.exp.num_episodes, env=env)
        end = perf_counter()
        print(f'Training took {end-start:.2f} seconds.')
        return history
    elif caller == 'jupyter':
        return agent.train(num_episodes=config.exp.num_episodes, env=env)


if __name__ == '__main__':
    main()
