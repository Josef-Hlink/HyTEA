import numpy as np
import wandb
from randomname import get_name
from hytea.utils import DotDict


# log hyperparameters to wandb
# at init intialize wandb run
# create random and unique id for each group and job_type

class WandbLogger:
    """ Log hyperparameters and results to wandb. 
    
    ### Args:
    `DotDict` config: full configuration of the experiment.
    """	
    def __init__(self, config: DotDict) -> None:
        self.config = config
        self.run = wandb.init(
            project = config.project_name,
            entity = config.wandb_team,
            group = config.group_name,
            job_type = config.job_type,
            config = config,
            reinit = True,
            monitor_gym = False,
        )
        return
    
    def log(self, data, step, commit=False) -> None:
        """ Log results to wandb. """
        self.run.log(data, step=step, commit=commit)
        return
    
    def log_config(self, config: dict) -> None:
        """ Log configuration to wandb. """
        self.run.config.update(config)
        return
    
    def update_summary(self, data: dict) -> None:
        """ Log summary to wandb. """
        self.run.summary.update(data)
        return
    
    def commit(self) -> None:
        """ Commit results to wandb. """
        self.run.log({}, commit=True)
        return
    
    def finish(self) -> bool:
        """ Finish wandb run. """	
        return self.run.finish()


def create_random_name() -> str:
    """ Create a random name for an individual run. """
    return get_name()

def create_group_name(config: DotDict, gen: int) -> str:
    """ Create a group name for a generation. """
    return f'Gen{gen}'

def create_project_name(config: DotDict) -> str:
    """ Create a project name for an experiment. """
    return f'{create_random_name()}-{config.env_name}-{config.num_train_episodes}-{config.num_test_episodes}-{config.num_runs}'

def create_job_type_name(bitstring: np.ndarray) -> str:
    """ Create a job type name for a bitstring. """
    return f'{create_random_name()}-{"".join(map(str, bitstring))}'
