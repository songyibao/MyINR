import math

import torch
from torch import nn

from src.models.layers import SineLayer
from src.configs.config import NetConfig, MyConfig
from src.models.layers import LayerRegistry
from src.utils.data_loader import ImageCompressionDataset
from src.utils.log import logger

class ConfigurableINRModel(nn.Module):
    def __init__(self, net_config: NetConfig, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        self.use_learnable_coords = net_config.use_learnable_coords
        if self.use_learnable_coords:
            dataset = ImageCompressionDataset(MyConfig.get_instance())
            coords, _, _, _, _ = dataset[0]
            self.table = nn.parameter.Parameter(1e-4 * (torch.rand((coords.shape[0], in_features)) * 2 - 1),
                                                requires_grad=True)
        # 判断是 in_features 参数是否传入实际值
        if in_features is None:
            raise ValueError("in_features 参数不能为空")
        if out_features is None:
            raise ValueError("out_features 参数不能为空")
        logger.info(f"模型初始化时输入维度: {in_features}")
        logger.info(f"模型初始化时输出维度: {out_features}")
        for layer_index, layer_config in enumerate(net_config.layers):
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


class ConfigurableBlockModelNew(nn.Module):
    def __init__(self, net_config: NetConfig, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        self.use_learnable_coords = net_config.use_learnable_coords
        if self.use_learnable_coords:
            dataset = ImageCompressionDataset(MyConfig.get_instance())
            coords, _, _, _, _ = dataset[0]
            self.table = nn.parameter.Parameter(1e-4 * (torch.rand((coords.shape[0], in_features)) * 2 - 1),
                                                requires_grad=True)
        # 判断是 in_features 参数是否传入实际值
        if in_features is None:
            raise ValueError("in_features 参数不能为空")
        if out_features is None:
            raise ValueError("out_features 参数不能为空")
        for layer_index, layer_config in enumerate(net_config.layers):
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
            else:
                layer_params['out_features'] = int(layer_params['out_features'] // math.sqrt(net_config.num_blocks))

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


class AuxModel_backup(nn.Module):
    def __init__(self, net_config: NetConfig, in_features, out_features):
        super().__init__()
        self.phase = 0
        self.in_features = in_features
        self.out_features = out_features
        self.other_layers = nn.ModuleList()
        self.use_aux_learnable_coords = net_config.use_aux_learnable_coords
        self.first_layer_out_features = net_config.layers[0].out_features
        # 判断是 in_features 参数是否传入实际值
        if in_features is None:
            raise ValueError("in_features 参数不能为空")
        if out_features is None:
            raise ValueError("out_features 参数不能为空")
        logger.info(f"模型初始化时输入维度: {in_features}")
        logger.info(f"模型初始化时输出维度: {out_features}")
        for layer_index, layer_config in enumerate(net_config.layers):
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
            self.other_layers.append(layer)
            if hasattr(layer, 'out_features'):
                in_features = layer.out_features
            elif hasattr(layer, 'out_channels'):
                in_features = layer.out_channels
            else:
                in_features = layer.out_features

        # self.layers = nn.Sequential(*self.layers)
        self.first_layer = self.other_layers[0]
        with torch.no_grad():
            self.first_layer.parameters()

        dataset = ImageCompressionDataset(MyConfig.get_instance())
        coords, _, _, _, _ = dataset[0]
        # self.table = nn.parameter.Parameter(torch.rand((coords.shape[0], self.first_layer_out_features)),
        #                                     requires_grad=True)
        # self.table = nn.parameter.Parameter((torch.rand((coords.shape[0],self.first_layer_out_features))*2 -1),requires_grad = True)
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((coords.shape[0],self.first_layer_out_features))*2 -1),requires_grad = True)
        print(self.table.shape)

        self.other_layers = self.other_layers[1:]
        self.other_layers = nn.Sequential(*self.other_layers)

    def forward(self, x):
        out = {}
        if self.phase == 0:
            # 第一阶段，训练除了第一层之外的所有层，第一层为table,可学习的坐标
            out['y'] = self.other_layers(self.table)
        elif self.phase == 1:
            # 第二阶段，训练第一层的输出接近训练好的table
            out['y1'] = self.first_layer(x)
            out['table'] = self.table
        elif self.phase == 2:
            # 推理阶段
            x = self.first_layer(x)
            out['y'] = self.other_layers(x)
        return out

def normalize_rows_to_minus_one_to_one(x, epsilon=1e-8):
    """
    将一个 (N, K) 的 tensor 按每行独立归一化到 (-1, 1) 范围。

    参数:
        x (torch.Tensor): 输入 tensor，形状为 (N, K)。
        epsilon (float): 防止除零的小数值，默认值为 1e-8。

    返回:
        torch.Tensor: 每行归一化到 (-1, 1) 的 tensor。
    """
    # 计算每行的最小值和最大值
    row_min, _ = x.min(dim=1, keepdim=True)  # 每行的最小值
    row_max, _ = x.max(dim=1, keepdim=True)  # 每行的最大值

    # 防止最大值和最小值相等导致除零
    denominator = row_max - row_min + epsilon

    # 按公式归一化到 (-1, 1)
    x_normalized = 2 * (x - row_min) / denominator - 1

    return x_normalized
class AuxModel(nn.Module):
    def __init__(self, in_features, out_features, mlp_width=128, mlp_depth=3, coords_shape=None):
        super().__init__()
        self.phase = 0
        self.in_features = in_features
        self.out_features = out_features
        # self.table = nn.Embedding(coords_shape[0], 8)
        self.table = nn.parameter.Parameter(1e-4*(torch.rand((coords_shape[0],8))*2 -1),requires_grad = True)
        self.first_layer = nn.Sequential(
            SineLayer(coords_shape[-1], mlp_width),
            SineLayer(mlp_width, mlp_width),
            SineLayer(mlp_width, mlp_width),
            nn.Linear(mlp_width, 8),
        )
        self.other_layers = nn.ModuleList()
        self.other_layers.append(SineLayer(8, mlp_width))
        for i in range(mlp_depth):
            self.other_layers.append(SineLayer(in_features=mlp_width, out_features=mlp_width))
        self.other_layers.append(nn.Linear(in_features=mlp_width, out_features=out_features))
        self.other_layers = nn.Sequential(*self.other_layers)
    def forward(self, x):
        out = {}
        if self.phase == 0:
            # 第一阶段，训练除了第一层之外的所有层，第一层为table,可学习的坐标
            out['y'] = self.other_layers(normalize_rows_to_minus_one_to_one(self.table))
        elif self.phase == 1:
            # 第二阶段，训练第一层的输出接近训练好的table
            out['y1'] = self.first_layer(x)
            # out['table'] = normalize_rows_to_minus_one_to_one(self.table)
        elif self.phase == 2:
            # 推理阶段
            y = self.first_layer(x)*0.0001
            # y = self.first_layer(x)
            out['y'] = self.other_layers(normalize_rows_to_minus_one_to_one(y))
        return out
