import toml
import torch
import torch.nn as nn
from typing import Dict, Any
from src.models.layers import LayerRegistry
from src.utils.log import logger


class ConfigurableINRModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any],in_features=None):
        super().__init__()

        self.layers = nn.ModuleList()
        # 判断是 in_features 参数是否传入实际值
        if in_features is None:
            logger.info(f'模型初始化时未传入输入维度,使用配置文件中的配置: {model_config["in_features"]}')
            in_features = model_config['in_features']
        else:
            logger.info(f'模型初始化时输入维度: {in_features}')
        for layer_config in model_config['layers']:
            layer_type = layer_config['type']
            layer_class = LayerRegistry.get(layer_type)
            if layer_class is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            layer_params = {k: v for k, v in layer_config.items() if k != 'type'}
            if 'in_features' not in layer_params and 'in_channels' not in layer_params:
                # 如果 layer_type 包含 'Conv' 则默认 in_channels=in_features
                if 'conv' in layer_type.lower():
                    layer_params['in_channels'] = in_features
                else:
                    layer_params['in_features'] = in_features

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