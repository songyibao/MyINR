import math

import numpy as np
from torch import nn, Tensor
import torch
class LayerRegistry:
    _layers = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_class):
            cls._layers[name] = layer_class
            return layer_class
        return decorator

    @classmethod
    def get(cls, name):
        return cls._layers.get(name)

@LayerRegistry.register('SynthesisLayer')
class SynthesisLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        """Instantiate a synthesis layer.

        Args:
            in_features (int): Input feature
            out_features (int): Output feature
            kernel_size (int): Kernel size
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            in_features,
            out_features,
            kernel_size
        )

        # More stable if initialized as a zero-bias layer with smaller variance
        # for the weights.
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data / out_features ** 2
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        # x = x.unsqueeze(0).unsqueeze(0).permute(2,3,0,1)
        # return self.conv_layer(x).permute(2,3,0,1).squeeze(0).squeeze(0)
        y = self.conv_layer(x)
        return y

@LayerRegistry.register('SynthesisResidualLayer')
class SynthesisResidualLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        """Instantiate a synthesis residual layer.

        Args:
            in_features (int): Input feature
            out_features (int): Output feature
            kernel_size (int): Kernel size
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features == out_features, \
            f'Residual layer in/out dim must match. Input = {in_features}, output = {out_features}'

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            in_features,
            out_features,
            kernel_size
        )

        # More stable if a residual is initialized with all-zero parameters.
        # This avoids increasing the output dynamic at the initialization
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data * 0.
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(0).unsqueeze(0).permute(2,3,0,1)
        return (self.conv_layer(x)+x).permute(2,3,0,1).squeeze(0).squeeze(0)
        # return self.conv_layer(self.pad(x)) + x

@LayerRegistry.register('SynthesisAttentionLayer')
class SynthesisAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        """Instantiate a synthesis attention layer.

        Args:
            in_features (int): Input feature
            out_features (int): Output feature
            kernel_size (int): Kernel size
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features == out_features, \
            f'Attention layer in/out dim must match. Input = {in_features}, output = {out_features}'

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))

        self.conv_layer_trunk = nn.Conv2d(
            in_features,
            out_features,
            kernel_size
        )
        self.conv_layer_sigmoid = nn.Conv2d(
            in_features,
            out_features,
            kernel_size
        )

        # Trunk is initialized as a residual block (i.e. all zero parameters)
        # Sigmoid branch is initialized as a linear layer (i.e. no bias and smaller variance
        # for the weights)
        with torch.no_grad():
            self.conv_layer_trunk.weight.data = self.conv_layer_trunk.weight.data * 0.
            self.conv_layer_trunk.bias.data = self.conv_layer_trunk.bias.data * 0.
            self.conv_layer_sigmoid.weight.data = self.conv_layer_sigmoid.weight.data / out_features ** 2

    def forward(self, x: Tensor) -> Tensor:
        # trunk = self.conv_layer_trunk(self.pad(x))
        trunk = self.conv_layer_trunk(x)
        weight = torch.sigmoid(self.conv_layer_sigmoid(self.pad(x)))
        return trunk * weight + x


@LayerRegistry.register('FourierReparamLinear')
class Fourier_reparam_linear(nn.Module):
    def __init__(self, in_features, out_features, high_freq_num=128, low_freq_num=128, phi_num=32, alpha=0.05):
        super(Fourier_reparam_linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num = high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha = alpha
        self.bases = self.init_bases()
        self.lamb = self.init_lamb()
        self.bias = nn.Parameter(torch.Tensor(self.out_features, 1), requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set = np.array([2 * math.pi * i / self.phi_num for i in range(self.phi_num)])
        high_freq = np.array([i + 1 for i in range(self.high_freq_num)])
        low_freq = np.array([(i + 1) / self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq) != 0:
            T_max = 2 * math.pi / low_freq[0]
        else:
            T_max = 2 * math.pi / min(high_freq)  # 取最大周期作为取点区间
        points = np.linspace(-T_max / 2, T_max / 2, self.in_features)
        bases = torch.Tensor((self.high_freq_num + self.low_freq_num) * self.phi_num, self.in_features)
        i = 0
        for freq in low_freq:
            for phi in phi_set:
                base = torch.tensor([math.cos(freq * x + phi) for x in points])
                bases[i, :] = base
                i += 1
        for freq in high_freq:
            for phi in phi_set:
                base = torch.tensor([math.cos(freq * x + phi) for x in points])
                bases[i, :] = base
                i += 1
        bases = self.alpha * bases
        bases = nn.Parameter(bases, requires_grad=False)
        return bases

    def init_lamb(self):
        self.lamb = torch.Tensor(self.out_features, (self.high_freq_num + self.low_freq_num) * self.phi_num)
        with torch.no_grad():
            m = (self.low_freq_num + self.high_freq_num) * self.phi_num
            for i in range(m):
                dominator = torch.norm(self.bases[i, :], p=2)
                self.lamb[:, i] = nn.init.uniform_(self.lamb[:, i], -np.sqrt(6 / m) / dominator,
                                                   np.sqrt(6 / m) / dominator)
        self.lamb = nn.Parameter(self.lamb, requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)

    def forward(self, x):
        weight = torch.matmul(self.lamb, self.bases)
        output = torch.matmul(x, weight.transpose(0, 1))
        output = output + self.bias.T
        return output


@LayerRegistry.register('LearnableEmbedding')
class LearnableEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embedding = nn.Embedding(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        return self.embedding(x)


@LayerRegistry.register('PosEncodingNeRF')
class PositionalEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.frequency_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.out_features = in_features * (2 * num_frequencies + 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (N, in_features) -> (N, in_features, 1)
        encodings = [x]
        for freq in self.frequency_bands:
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1).view(x.shape[0], -1)

@LayerRegistry.register('Linear')
class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        return self.linear(x)
@LayerRegistry.register('FinalLinear')
class FinalLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).squeeze(0)
        return self.linear(x)
@LayerRegistry.register('Conv1d')
class Conv1dLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        return self.conv(x)

@LayerRegistry.register('Conv2d')
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        return self.conv(x)

@LayerRegistry.register('ReLU')
class ReLULayer(nn.Module):
    def __init__(self,in_features: int, out_features: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        return self.relu(x)

@LayerRegistry.register('LeakyReLU')
class LeakyReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,negative_slope: float = 0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        return self.leaky_relu(x)

@LayerRegistry.register('SineLayer')
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate