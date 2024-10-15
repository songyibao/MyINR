import math

import torch
from kan.KANLayer import *

from src.configs.config import NetConfig, MyConfig
from src.models.layers import LayerRegistry
from src.utils.data_loader import ImageCompressionDataset
from src.utils.log import logger


class ConfigurableINRModel(nn.Module):
    def __init__(self, net_config: NetConfig, in_features,out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        self.use_learnable_coords = net_config.use_learnable_coords
        if self.use_learnable_coords:
            dataset = ImageCompressionDataset(MyConfig.get_instance())
            coords,_,_,_,_ = dataset[0]
            self.table = nn.parameter.Parameter(1e-4 * (torch.rand((coords.shape[0],in_features))*2 -1),requires_grad = True)
        # 判断是 in_features 参数是否传入实际值
        if in_features is None:
            raise ValueError("in_features 参数不能为空")
        if out_features is None:
            raise ValueError("out_features 参数不能为空")
        logger.info(f"模型初始化时输入维度: {in_features}")
        logger.info(f"模型初始化时输出维度: {out_features}")
        for layer_index,layer_config in enumerate(net_config.layers):
            layer_type = layer_config.type
            layer_class = LayerRegistry.get(layer_type)
            if layer_class is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            if layer_config.in_features is None:
                layer_config.in_features = in_features

            items = layer_config.model_dump(exclude_unset=True).items()
            layer_params = {k: v for k, v in items if k != 'type'}

            if layer_index == len(net_config.layers) - 1:
                layer_params['out_features'] = out_features

            layer = layer_class(**layer_params)
            self.layers.append(layer)
            if hasattr(layer, 'out_features'):
                in_features = layer.out_features
            elif hasattr(layer, 'out_channels'):
                in_features = layer.out_channels
            else:
                in_features = layer.out_features

        self.layers = nn.Sequential(*self.layers)





    def forward(self, x):
        if self.use_learnable_coords:
            x = self.table
        y = self.layers(x)
        if self.layers[0].__class__.__name__ == 'ComplexGaborLayer':
            y = y.real
        return y

class ConfigurableStackedModel(nn.Module):
    def __init__(self, net_config: NetConfig, in_features,out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nets = nn.ModuleList()
        self.total_out_features = 0
        for i in range(self.out_features):
            net = nn.ModuleList()
            for layer_index,layer_config in enumerate(net_config.layers):
                layer_type = layer_config.type
                layer_class = LayerRegistry.get(layer_type)
                if layer_class is None:
                    raise ValueError(f"Unsupported layer type: {layer_type}")

                if layer_config.in_features is None:
                    layer_config.in_features = in_features

                items = layer_config.model_dump(exclude_unset=True).items()
                layer_params = {k: v for k, v in items if k != 'type'}

                layer_params['out_features'] = int(layer_params['out_features']/math.sqrt(out_features))
                self.bottle_width = layer_params['out_features']
                self.total_out_features = self.bottle_width * out_features

                layer = layer_class(**layer_params)
                net.append(layer)
                if hasattr(layer, 'out_features'):
                    in_features = layer.out_features
                elif hasattr(layer, 'out_channels'):
                    in_features = layer.out_channels
                else:
                    in_features = layer.out_features
            linear_layer = nn.Linear(self.bottle_width, 1)
            with torch.no_grad():
                linear_layer.weight.uniform_(-1/self.bottle_width,1/self.bottle_width)
            net.append(linear_layer)
            self.nets.append(nn.Sequential(*net))
        self.nets = nn.ParameterList(self.nets)
    def initialize_weights(self):
        # 创建一个新的权重矩阵
        weight_shape = (self.out_features, self.bottle_width * self.out_features)
        weights = torch.empty(weight_shape)

        # 对于每个线性层的部分进行初始化
        for i in range(self.out_features):
            # 定义当前层的权重部分
            start_index = i * self.bottle_width
            end_index = (i + 1) * self.bottle_width
            # 使用均匀分布初始化当前部分的权重
            weights[:, start_index:end_index] = torch.empty(self.out_features, self.bottle_width).uniform_(-1/self.bottle_width, 1/self.bottle_width)

        # 将初始化后的权重赋值给 final_linear
        self.final_linear.weight.data = weights

        # 可选：初始化偏置为零
        with torch.no_grad():
            self.final_linear.bias.zero_()

    def forward(self, x):
        output = []
        for i in range(self.out_features):
            output.append(self.nets[i](x))
        y = torch.cat(output, dim=-1)
        if self.nets[0][0].__class__.__name__ == 'ComplexGaborLayer':
            y = y.real
        return y

class ConfigurableBlockModel(nn.Module):
    def __init__(self, net_config: NetConfig, in_features,out_features,input_size):
        super().__init__()
        self.num_blocks = net_config.num_blocks
        self.input_size = input_size
        self.block_size = self.input_size // self.num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.nets = nn.ModuleList()
        for i in range(self.num_blocks):
            net = nn.ModuleList()
            for layer_index,layer_config in enumerate(net_config.layers):
                layer_type = layer_config.type
                layer_class = LayerRegistry.get(layer_type)
                if layer_class is None:
                    raise ValueError(f"Unsupported layer type: {layer_type}")

                if layer_config.in_features is None:
                    layer_config.in_features = in_features

                items = layer_config.model_dump(exclude_unset=True).items()
                layer_params = {k: v for k, v in items if k != 'type'}

                if layer_index != len(net_config.layers) - 1:
                    layer_params['out_features'] = int(layer_params['out_features'] / math.sqrt(self.num_blocks))

                layer = layer_class(**layer_params)
                net.append(layer)
                if hasattr(layer, 'out_features'):
                    in_features = layer.out_features
                elif hasattr(layer, 'out_channels'):
                    in_features = layer.out_channels
                else:
                    in_features = layer.out_features
            self.nets.append(nn.Sequential(*net))

        self.nets = nn.ParameterList(self.nets)
    def forward(self, x):
        output = []
        for i in range(self.num_blocks):
            if i == self.num_blocks - 1:
                output.append(self.nets[i](x[i*self.block_size:,:]))
            else:
                output.append(self.nets[i](x[i*self.block_size:(i+1)*self.block_size,:]))
        y = torch.cat(output, dim=0)
        if self.nets[0][0].__class__.__name__ == 'ComplexGaborLayer':
            y = y.real
        return y