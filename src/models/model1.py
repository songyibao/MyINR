import toml
import torch
import torch.nn as nn
from typing import Dict, Any

from src.configs.config import NetConfig
from src.models.layers import LayerRegistry
from src.utils.log import logger


class ConfigurableINRModel(nn.Module):
    def __init__(self, net_config: NetConfig, in_features,out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
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




    def forward(self, x):
        for layer in self.layers:
            logger.debug(f"Layer: {layer}")
            logger.debug(f'Input shape: {x.shape}')
            x = layer(x)
            logger.debug(f'Output shape: {x.shape}')

            logger.debug(f'Next in_features: {layer.out_features if hasattr(layer, "out_features") else layer.out_channels}')
        return x


def load_model_from_config(config_path: str) -> ConfigurableINRModel:
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return ConfigurableINRModel(config)