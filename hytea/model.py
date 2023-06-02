import torch
import torch.nn.functional as F

    
class Model(torch.nn.Module):
    
    def __init__(self, 
            input_size: int, output_size: int, hidden_size: int, 
            hidden_activation: str, num_layers: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.using_dropout = dropout_rate > 0
        self.hidden_activation = eval(f'F.{hidden_activation}')

        self.fc0 = torch.nn.Linear(input_size, hidden_size)
        self.do0 = torch.nn.Dropout(dropout_rate) if self.using_dropout else None

        for i in range(num_layers-1):
            setattr(self, f'fc{i+1}', torch.nn.Linear(hidden_size, hidden_size))
            setattr(self, f'do{i+1}', torch.nn.Dropout(dropout_rate) if self.using_dropout else None)
        
        self.actor_head = torch.nn.Linear(hidden_size, output_size)
        self.critic_head = torch.nn.Linear(hidden_size, 1)
        
        return
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the model.

        Takes a state (or batch of states) and returns the actor and critic outputs.
        """
        for i in range(0, self.num_layers):
            x = getattr(self, f'fc{i}')(x)
            if self.using_dropout:
                x = getattr(self, f'do{i}')(x)
            x = self.hidden_activation(x)

        return F.softmax(self.actor_head(x), dim=-1), self.critic_head(x)
    
    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """ For more accurate type hinting. """
        return super().__call__(*args, **kwargs)
