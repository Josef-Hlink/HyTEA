import torch


class Trajectory:
    """ A set of transitions sampled from one episode. """

    def __init__(self, state_shape: tuple, max_length: int, device) -> None:
        """ Initializes a trajectory. """
        self.device = device
        self.S = torch.empty((max_length, *state_shape), dtype=torch.float32, device=self.device)
        self.A = torch.empty(max_length, device=self.device)
        self.R = torch.empty(max_length, dtype=torch.float32, device=self.device)
        self.S_ = torch.empty((max_length, *state_shape), dtype=torch.float32, device=self.device)
        self.D = torch.empty(max_length, dtype=torch.bool, device=self.device)
        self.ml = max_length
        self.l = 0

    def add(self, s: torch.Tensor, a: torch.Tensor, r: float, s_: torch.Tensor, d: bool) -> None:
        """ Adds a transition to the trajectory. """
        if self.full:
            raise RuntimeError('Trajectory is full.')
        self.S[self.l] = s
        self.A[self.l] = a
        self.R[self.l] = torch.tensor(r, dtype=torch.float32)
        self.S_[self.l] = s_
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
        return map(lambda x: x[:self.l], (self.S, self.A, self.R, self.S_, self.D))

    def __len__(self) -> int:
        """ Returns the number of transitions in the trajectory. """
        return self.l
