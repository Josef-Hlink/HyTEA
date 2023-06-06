import torch
import torch.nn.functional as F


class Model(torch.nn.Module):

    def __init__(self,
        input_size: int, output_size: int, hidden_size: int,
        num_layers: int, dropout_rate: float, hidden_activation: str, output_activation: str
    ) -> None:
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_activation = {'relu': F.relu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}[hidden_activation]
        self.output_activation = {'relu': F.relu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}[output_activation]

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        
        if dropout_rate > 0:
            self.do1 = torch.nn.Dropout(dropout_rate)
        
        for i in range(num_layers):
            setattr(self, f'fc{i+2}', torch.nn.Linear(hidden_size, hidden_size))
            if dropout_rate > 0:
                setattr(self, f'do{i+2}', torch.nn.Dropout(dropout_rate))
        
        setattr(self, f'fc{num_layers+2}', torch.nn.Linear(hidden_size, output_size))
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.dropout_rate > 0:
            x = self.do1(x)
        x = self.hidden_activation(x)
        for i in range(self.num_layers):
            x = getattr(self, f'fc{i+2}')(x)
            if self.dropout_rate > 0:
                x = getattr(self, f'do{i+2}')(x)
            x = self.hidden_activation(x)
        x = getattr(self, f'fc{self.num_layers+2}')(x)
        return self.output_activation(x)
