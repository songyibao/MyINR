import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.frequency_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (N, in_features) -> (N, in_features, 1)
        encodings = [x]
        for freq in self.frequency_bands:
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1).view(x.shape[0], -1)

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.linear.weight, -1 / in_features, 1 / in_features)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class INRModel(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, num_frequencies=10, omega_0=30):
        super().__init__()
        self.pos_encoding = PositionalEncoding(in_features, num_frequencies)
        encoded_features = in_features * (2 * num_frequencies + 1)

        self.net = nn.ModuleList([
            SIRENLayer(encoded_features, hidden_features, omega_0)
        ])

        for _ in range(hidden_layers):
            self.net.append(SIRENLayer(hidden_features, hidden_features, omega_0))

        self.net.append(nn.Linear(hidden_features, out_features))

    def forward(self, x):
        x = self.pos_encoding(x)
        for layer in self.net[:-1]:
            x = layer(x)
        x = self.net[-1](x)
        return x

# 使用示例
# model = INRModel(in_features=2, hidden_features=256, hidden_layers=3, out_features=3)