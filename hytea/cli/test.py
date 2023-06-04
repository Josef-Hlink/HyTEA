from time import perf_counter
import argparse
from pathlib import Path

from hytea.utils import DotDict
from hytea import Environment, Model, Agent
from hytea.utils.wblog import WandbLogger, create_random_name

from yaml import safe_load
import torch


def test(args: argparse.Namespace) -> None:
    """ Tests the training process of a single agent. """

    with open(Path(__file__).resolve().parents[1] / 'config.yaml', 'r') as f:
        blueprint = DotDict.from_dict(safe_load(f))

    device = torch.device('cpu')
    env = Environment(env_name=args.env_name, device=device)
    model = Model(
        input_size = env.observation_space.shape[0],
        output_size = env.action_space.n,
        hidden_size = blueprint.network.hidden_size.default,
        hidden_activation = blueprint.network.hidden_activation.default,
        num_layers = blueprint.network.num_layers.default,
        dropout_rate = blueprint.network.dropout_rate.default,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=blueprint.optimizer.lr.default)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=blueprint.optimizer.lr_step.default, gamma=blueprint.optimizer.lr_decay.default)
    agent = Agent(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        gamma = blueprint.agent.gamma.default,
        ent_reg_weight = blueprint.agent.ent_reg_weight.default,
        bl_sub = blueprint.agent.bl_sub.default,
        device = device
    )

    for _ in range(args.num_runs):
        if args.use_wandb:
            logger = WandbLogger(
                project_name='test',
                wandb_team=args.wandb_team,
                group_name=args.group_name,
                job_type=create_random_name(),
                config = {},
            )
            
        start = perf_counter()
        history = agent.train(num_episodes=args.num_train_episodes, env=env)
        
        # log the history
        for i, h in enumerate(history):
            logger.log({'train_reward': h}, step=i)
            
        print(f'Training took {perf_counter() - start:.2f} seconds.')
        test_reward = agent.test(num_episodes=args.num_test_episodes)
        print(f'Test reward: {test_reward:.2f}')
        
        if args.use_wandb: 
            logger.update_summary({'test_reward': test_reward})
            logger.finish()

    return

def add_test_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--env', dest='env_name', type=str,
        default='LunarLander-v2', help='The environment to use.'
    )
    parser.add_argument('--train', dest='num_train_episodes', type=int,
        default=1000, help='The number of episodes to train for.'
    )
    parser.add_argument('--test', dest='num_test_episodes', type=int,
        default=100, help='The number of episodes to test for.'
    )
    parser.add_argument('--runs', dest='num_runs', type=int,
        default=1, help='The number of individual runs to average over.'
    )
    parser.add_argument('-W', '--use-wandb', dest='use_wandb', action='store_true',
        help='Whether to use wandb for logging.'
    )
    parser.add_argument('--wt', dest='wandb_team', type=str,
        default='hytea', help='The name of the wandb team.'
    )
    parser.add_argument('--wtg', dest='group_name', type=str,
        default='test', help='The name of the wandb team group.'
    )
    return parser
