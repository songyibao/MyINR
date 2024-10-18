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




@LayerRegistry.register('LinearRelu')
class LinearRelu(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
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


@LayerRegistry.register('ReLU')
class ReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.relu(x)


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


@LayerRegistry.register('FinerLayer')
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.,
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out


@LayerRegistry.register('FinerOutmostLinear')
class FinerOutmostLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0,
                                        np.sqrt(6 / in_features) / omega_0)

    def forward(self, x):
        return self.linear(x)


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
            self.l_omega = Parameter(torch.tensor(0.5), requires_grad=True)

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
            res = torch.exp(1 - x)
        else:
            res = torch.sin(x)
        return res

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega * self.linear(input)
        return torch.sin(intermediate), intermediate


@LayerRegistry.register('GaussLayer')
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.scale = torch.tensor(scale)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


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


