import wandb
import randomname
import numpy as np
from hytea.utils import DotDict


# log hyperparameters to wandb
# at init intialize wandb run
# create random and unique id for each group and job_type

class WandbLogger:
    """ Log hyperparameters and results to wandb. 
    
    Args:
        config (dict): configuration dictionary.
    """	
    def __init__(self, config: DotDict) -> None:
        self.config = config
        self.run = wandb.init(
            project = config.project_name,
            entity = config.wandb_team,
            group = config.group_name,
            job_type = config.job_type,
            config = config
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
    
# create random name for group and job_type
def create_random_name() -> str:
    return randomname.get_name()

# a group represents the entire experiment and can be versioned on wandb
def create_group_name(config: DotDict, gen: int) -> str:
    return f'Gen{gen}'

def create_project_name(config: DotDict) -> str:
    return f'{create_random_name()}-{config.env_name}-{config.num_train_episodes}-{config.num_test_episodes}-{config.num_runs}'

def create_job_type_name(bitstring: np.ndarray) -> str:
    return f'{create_random_name()}-{"".join(map(str, bitstring))}'
