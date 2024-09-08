import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.configs.config import GlobalConfig, ModelConfig, TrainConfig, SaveConfig
from src.decompress import decompress_and_save
from src.models.inputs import periodic_encoding, positional_encoding, get_coordinate_grid
from src.models.model1 import ConfigurableINRModel
from src.train import train_inr
from src.utils.data_loader import ImageCompressionDataset
from src.utils.device import global_device
from src.utils.log import logger


def test(model_config:ModelConfig, train_config:TrainConfig, save_config:SaveConfig,deivce=global_device):
    logger.info(f'模型配置:{model_config.config}')
    logger.info("加载和预处理图像")
    dataset = ImageCompressionDataset(train_config.image_path)
    logger.info("创建坐标网格")
    coords, pixels = dataset[0]
    logger.info("位置编码")
    coords = positional_encoding(coords,num_frequencies=model_config.num_frequencies)
    logger.info(f"位置编码后的形状{coords.shape}")
    coords, pixels = coords.to(deivce), pixels.to(deivce)
    target_image = pixels.view(dataset.h,dataset.w,3).to(device)
    inr_model = ConfigurableINRModel(model_config.config,in_features=coords.shape[-1])
    # 训练模型并记录过程
    trained_inr_model = train_inr(model_input=coords, target_image=target_image, model=inr_model, device=device, config=train_config)

    # 保存模型
    logger.info("保存模型")
    if not os.path.exists(save_config.base_output_path):
        os.makedirs(save_config.base_output_path)

    torch.save(trained_inr_model.state_dict(), os.path.join(save_config.model_save_path,save_config.model_name).__str__())

    # 记录模型保存路径
    logger.info("保存模型到wandb")

    # 创建坐标网格
    # h, w, _ = img_np.shape

    # 解压并保存图像
    logger.info("重建并保存图像")
    # coords = get_coordinate_grid(h, w, torch.device('cpu'))
    # coords = positional_encoding(coords)
    model = ConfigurableINRModel(model_config.config,in_features=coords.shape[-1])
    model.load_state_dict(torch.load(os.path.join(save_config.model_save_path,save_config.model_name).__str__(),weights_only=True,map_location=device))
    decompress_and_save(inr_model=model, model_input=coords, base_output_path=save_config.base_output_path, config=global_config)

    # 保存生成的图像到wandb
    logger.info("保存生成的图像到wandb")


device = global_device
global_config = GlobalConfig()
# logger.info(f'{global_config}')
# 以下三个都是类的实例化，不是字典
global_model_config = global_config.model_config
global_train_config = global_config.train_config
global_save_config = global_config.save_config
# out_features_tuple = (32,64,128,256)
# for i in out_features_tuple:
#     num_layers = len(global_model_config.config['layers'])
#     for index,layer_config in enumerate(global_model_config.config['layers']):
#         if index != num_layers-1:
#             layer_config['out_features'] = i
#     logger.info(global_config.config)

test(global_model_config, global_train_config, global_save_config)



