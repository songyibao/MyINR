import math

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn.parameter import Parameter

from src.configs.config import MyConfig


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


@LayerRegistry.register('KAN')
class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


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
        x = x.unsqueeze(0).unsqueeze(0).permute(2, 3, 0, 1)
        return (self.conv_layer(x) + x).permute(2, 3, 0, 1).squeeze(0).squeeze(0)
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


@LayerRegistry.register('FourierFeatureMapping')
class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, out_features,omega_0=30.):
        super(FourierFeatureMapping, self).__init__()
        self.omega_0 = omega_0
        self.B = nn.Parameter(torch.ones((in_features, int(out_features / 2))), requires_grad=True)
        self.B.data.uniform_(-math.sqrt(6 / in_features) / omega_0, math.sqrt(6 / in_features) / omega_0)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x_proj = x @ self.B * self.omega_0 *self.omega_0
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
@LayerRegistry.register('LearnablePE')
class LearnablePositionEncoding(nn.Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        super(LearnablePositionEncoding, self).__init__()
        # 定义线性层，将输入维度从 k 映射到 l
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x 的形状为 [N, k]
        # 通过线性层进行变换
        embedded = self.linear(x)
        return embedded
@LayerRegistry.register('LearnableEmbedding')
class LearnableEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        config = MyConfig.get_instance()
        self.h, self.w = config.train.h, config.train.w
        num_embeddings = self.h * self.w
        self.embedding = nn.Embedding(num_embeddings, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.embedding(x)


@LayerRegistry.register('Embedding')
class Embedding(nn.Module):
    def __init__(self, in_features, num_frequencies, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = num_frequencies
        self.in_channels = in_features
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_features * (len(self.funcs) * num_frequencies + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


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
    def __init__(self, in_features: int, out_features: int, need_manual_init: bool = False,
                 hidden_omega_0: float = 60., use_cfloat_dtype: bool = False, use_relu: bool = False):
        super().__init__()
        data_type = torch.float if not use_cfloat_dtype else torch.cfloat
        linear = nn.Linear(in_features, out_features, dtype=data_type)
        if need_manual_init and use_cfloat_dtype:
            raise ValueError('WIRE(use_cfloat_dtype) and SIREN(need_manual_init) cannot be used together')
        if need_manual_init is True:
            with torch.no_grad():
                linear.weight.uniform_(-np.sqrt(6 / in_features) / hidden_omega_0,
                                       np.sqrt(6 / in_features) / hidden_omega_0)
        if use_relu:
            self.net = nn.Sequential(
                linear,
                nn.ReLU()
            )
        else:
            self.net = linear
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.net(x)


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


@LayerRegistry.register('Conv2dPE')
class HierarchicalPositionalEncoding(nn.Module):
    def __init__(self, in_features=2, base_dim=16, out_features=32, num_layers=4):
        super(HierarchicalPositionalEncoding, self).__init__()
        self.initial_layer = nn.Linear(in_features, base_dim)
        self.layers = nn.ModuleList([nn.Linear(base_dim * 2 ** i, base_dim * 2 ** (i + 1)) for i in range(num_layers)])
        self.final_layer = nn.Linear(base_dim * 2 ** num_layers, out_features)
        self.activation = nn.ReLU()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, coords):
        z = self.activation(self.initial_layer(coords))
        for layer in self.layers:
            z = self.activation(layer(z))
        return self.final_layer(z)


@LayerRegistry.register('ReLU')
class ReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.relu(x)


@LayerRegistry.register('LeakyReLU')
class LeakyReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, negative_slope: float = 0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.leaky_relu(x)


@torch.jit.script
def bspline_wavelet(x, scale):
    return (1 / 6) * F.relu(scale * x) \
        - (8 / 6) * F.relu(scale * x - (1 / 2)) \
        + (23 / 6) * F.relu(scale * x - (1)) \
        - (16 / 3) * F.relu(scale * x - (3 / 2)) \
        + (23 / 6) * F.relu(scale * x - (2)) \
        - (8 / 6) * F.relu(scale * x - (5 / 2)) \
        + (1 / 6) * F.relu(scale * x - (3))


@LayerRegistry.register('BWNonlin')
class BSplineWavelet(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale=torch.as_tensor(1)):
        super().__init__()
        self.scale = torch.as_tensor(scale)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        output = bspline_wavelet(x, self.scale)

        return output


# Sigmod层
@LayerRegistry.register('Sigmoid')
class SigmoidLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.sigmoid(x)


@LayerRegistry.register('LeakyReLU')
class LeakyReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, negative_slope: float = 0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.leaky_relu(x)


class FreqFactor(nn.Module):
    def __init__(self, dims: int, omega=60, ):
        super().__init__()
        self.omega = Parameter(torch.Tensor(dims))
        # 所有值初始化为 omega
        self.omega.data.fill_(omega)

    def forward(self):
        return self.omega


@LayerRegistry.register('SineLayer')
class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=60, enable_learnable_omega=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.omega = omega_0
        self.learnable_omegas = None
        self.enable_learnable_omega = enable_learnable_omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

        if self.enable_learnable_omega:
            self.l_omega = FreqFactor(out_features, omega=self.omega)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                            np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        res = None
        x = self.linear(input)
        if self.enable_learnable_omega:
            factors = self.l_omega()
            res = torch.sin(factors.mul(x))
        else:
            res = torch.sin(self.omega * x)
        return res

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega * self.linear(input)
        return torch.sin(intermediate), intermediate


class SinCosActivation(nn.Module):
    def __init__(self, num_sinusoids=1, omega_0=60):
        super(SinCosActivation, self).__init__()
        self.omega_0 = omega_0
        self.num_sinusoids = num_sinusoids

        # 可学习的振幅权重，并进行归一化
        self.amplitudes_sin = nn.Parameter(torch.randn(num_sinusoids), requires_grad=True)
        self.amplitudes_cos = nn.Parameter(torch.randn(num_sinusoids), requires_grad=True)

        # 每个正弦函数和余弦函数的频率初始化为 0.1 到 1.0 的值
        self.frequencies = nn.Parameter(torch.ones(num_sinusoids), requires_grad=True)
        # self.frequencies.data.uniform_(0.1, 1.0)

        # 每个正弦函数和余弦函数的相位偏移 (可学习的)
        self.phases_sin = nn.Parameter(torch.randn(num_sinusoids))
        self.phases_cos = nn.Parameter(torch.randn(num_sinusoids))

    def forward(self, x):
        # 扩展 x 的维度使得与 sinusoids 的维度一致
        x_expanded = x.unsqueeze(-1)  # 形状 (batch_size, input_size, 1)

        # 计算 sinusoids
        sinusoids = torch.sin(self.omega_0 * self.frequencies * x_expanded + self.phases_sin)
        # 计算 cosusoids
        cosusoids = torch.cos(self.omega_0 * self.frequencies * x_expanded + self.phases_cos)

        # 归一化振幅权重
        amplitudes_sin_normalized = torch.softmax(self.amplitudes_sin, dim=0)
        amplitudes_cos_normalized = torch.softmax(self.amplitudes_cos, dim=0)

        # 加权求和 sin 和 cos 的输出
        output_sin = torch.einsum('ijk,k->ij', sinusoids, amplitudes_sin_normalized)
        output_cos = torch.einsum('ijk,k->ij', cosusoids, amplitudes_cos_normalized)

        # 返回 sin 和 cos 的组合
        return output_sin + output_cos


@LayerRegistry.register('ExpLayer')
class ExpLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=60, enable_learnable_omega=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.omega = omega_0
        self.learnable_omegas = None
        self.enable_learnable_omega = enable_learnable_omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

        if self.enable_learnable_omega:
            self.l_omega = Parameter(torch.tensor(0.5),requires_grad=True)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features),
                                            np.sqrt(6 / self.in_features))

    def forward(self, input):
        res = None
        x = self.linear(input)
        if self.enable_learnable_omega:
            res = torch.exp(1-x)
        else:
            res = torch.sin(x)
        return res

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega * self.linear(input)
        return torch.sin(intermediate), intermediate


@LayerRegistry.register('GaussLayer')
class GaussLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with Gaussian non linearity
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=20, scale=30.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.scale = torch.tensor(scale)
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return torch.exp(-(self.scale * self.linear(input)) ** 2)


@LayerRegistry.register('ComplexGaborLayer')
class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=20, sigma0=30.0,
                 trainable=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


@LayerRegistry.register('StackLayer')
class StackLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.layers = nn.ModuleList()
        for i in range(out_features):
            layer = nn.Linear(in_features, 1)
            with torch.no_grad():
                layer.weight.random_(0, 1)
            self.layers.append(layer)

    def forward(self, x):
        outputs = []  # 用来收集每个 net 的输出
        # 遍历所有的子网络
        for net in self.layers:
            out = net(x)  # 获取每个 net 的输出，形状是 [N, 1]
            outputs.append(out)  # 将输出添加到列表中

        # 将所有输出沿着 dim=1 进行拼接，输出的形状将是 [N, numNet]
        res = torch.cat(outputs, dim=1)

        return res


@LayerRegistry.register('LearnablePositionalEncoding')
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.embedding = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        return x + self.embedding
