import torch
import torch.nn as nn
import torch.nn.functional as F

# Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, output_size)

        # Initialize weights
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
        nn.init.kaiming_normal_(self.output.weight)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
        return self.output(x)


# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_layers - 1):
            self.feature_layers.append(nn.Linear(hidden_size, hidden_size))

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Initialize weights
        for layer in self.feature_layers:
            nn.init.kaiming_normal_(layer.weight)

        for module in self.value_stream:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

        for module in self.advantage_stream:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x):
        for layer in self.feature_layers:
            x = F.leaky_relu(layer(x), 0.1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(1, keepdim=True)
