import torch


class Trajectory:
    """ A set of transitions sampled from one episode. """

    def __init__(self, max_length: int, device) -> None:
        """ Initializes a trajectory. """
        self.device = device
        self.P = torch.empty(max_length, dtype=torch.float32, device=self.device)
        self.R = torch.empty(max_length, dtype=torch.float32, device=self.device)
        self.V = torch.empty(max_length, dtype=torch.float32, device=self.device)
        self.E = torch.empty(max_length, dtype=torch.float32, device=self.device)
        self.D = torch.empty(max_length, dtype=torch.bool, device=self.device)
        self.ml = max_length
        self.l = 0

    def add(self, p: torch.Tensor, v: torch.Tensor, r: float, e: torch.Tensor, d: bool) -> None:
        """ Adds a transition to the trajectory. """
        if self.full:
            raise RuntimeError('Trajectory is full.')
        self.P[self.l] = p
        self.R[self.l] = torch.tensor(r, dtype=torch.float32)
        self.V[self.l] = v
        self.E[self.l] = e
        self.D[self.l] = torch.tensor(d, dtype=torch.bool)
        self.l += 1

    @property
    def full(self) -> bool:
        """ Returns true if the trajectory is full. """
        return self.l == self.ml
    
    @property
    def total_reward(self) -> float:
        """ Returns the total reward of the trajectory. """
        return self.R[:self.l].sum().item()

    def unpack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Returns the trajectory as a tuple of tensors. """
        return map(lambda x: x[:self.l], (self.P, self.R, self.V, self.E, self.D))
    
    def __len__(self) -> int:
        """ Returns the number of transitions in the trajectory. """
        return self.l
