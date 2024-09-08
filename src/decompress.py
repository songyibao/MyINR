import datetime
import os

import numpy as np
import toml
import torch
import platform

from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor

from src.utils.device import global_device

os_type = platform.system()
if os_type == 'Windows':
    pass
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.models.inputs import get_coordinate_grid, positional_encoding
from src.models.model1 import ConfigurableINRModel
from src.configs.config import GlobalConfig
from src.utils.evaluate import get_original_image_numpy, calculate_bpp, evaluate_ndarray, evaluate_tensor, \
    evaluate_tensor_h_w_3
from src.utils.log import logger
import matplotlib.pyplot as plt

def create_experiment_directory(base_path: str) -> str:
    """创建实验目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_path, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_config_to_toml(config: dict, file_path: str):
    """将配置保存到.toml文件"""
    try:
        with open(file_path, 'w') as toml_file:
            toml.dump(config, toml_file)
    except Exception as e:
        logger.error(f"保存配置文件时发生错误: {e}")

def save_experiment_summary(summary: dict, file_path: str):
    """保存实验摘要"""
    with open(file_path, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

def create_comparison_image(original_image, reconstructed_image, save_path: str):
    """创建原始图像和重建图像的比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(reconstructed_image)
    ax2.set_title("Reconstructed Image")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def decompress_and_save(inr_model, model_input, config: GlobalConfig, base_output_path: str):
    experiment_dir = create_experiment_directory(base_output_path)
    global_config_dict = config.config
    model_config_dict = config.model_config.config
    original_image_path = config.train_config.image_path
    # 获取原始图像并保存
    original_image = get_original_image_numpy(original_image_path)
    original_image_path = os.path.join(experiment_dir, 'original_image.png')
    to_pil_image(original_image).save(original_image_path)

    logger.info(f"图像形状{original_image.shape}")
    shape = original_image.shape

    # 使用CPU模式并设置模型为评估模式
    inr_model = inr_model.to('cpu').eval()
    model_input = model_input.to('cpu')

    # 计算模型输出
    logger.info("计算模型输出")
    with torch.no_grad():
        # [h * w, 3] -> [h, w, 3]
        pixels = inr_model(model_input).view(shape[0], shape[1], 3)

    # 计算评估指标
    logger.info("计算原图像和重建图像psnr和ssim")

    # Converts a PIL Image or numpy. ndarray (H x W x C) in the range [0, 255] to a torch. FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    trans_fun = v2.Compose([v2.ToImage(), v2.ToDtype(torch. float32, scale=True)])
    # [H,W,3] -> [3,H,W]
    x = trans_fun(original_image)
    # [3,H,W] -> [H,W,3]
    x = x.permute(1,2,0)

    result=evaluate_tensor_h_w_3(x, torch.clamp(pixels, 0, 1))
    # 计算bpp
    logger.info("计算 bpp")
    bpp = calculate_bpp(torch.tensor(original_image),inr_model)
    result.update({
        "bpp": bpp
    })
    logger.info(f'{result}')
    # 转换并保存图像
    logger.info("转换和保存图像")
    reconstructed_image = to_pil_image(pixels.permute(2, 0, 1)) # Tensor 类型会被内部 permute
    img_save_path = os.path.join(experiment_dir, 'reconstructed_image.png')
    reconstructed_image.save(img_save_path)



    # 创建比较图像
    comparison_image_path = os.path.join(experiment_dir, 'comparison.png')
    create_comparison_image(original_image, reconstructed_image, comparison_image_path)






    # 保存评估指标
    result_file_path = os.path.join(experiment_dir, 'evaluation_results.toml')
    save_config_to_toml(result, result_file_path)

    # 保存配置文件
    config_file_path = os.path.join(experiment_dir, 'config.toml')
    save_config_to_toml(global_config_dict, config_file_path)



    # 创建并保存实验摘要
    summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Model Configuration": model_config_dict,
        "Evaluation Results": result,
        "Original Image": original_image_path,
        "Reconstructed Image": img_save_path,
        "Comparison Image": comparison_image_path
    }

    summary_path = os.path.join(experiment_dir, 'experiment_summary.txt')
    save_experiment_summary(summary, summary_path)

    logger.info(f'实验结果已保存到目录: {experiment_dir}')
    return experiment_dir





# 使用示例
# compress('data/raw/image.jpg', 'models/inr_model.pth', 'compressed_image.pth')
# Load and preprocess the image
if os.getenv('MODE',"TRAIN").upper()=="SINGLE":
    global_config = GlobalConfig()
    train_config = global_config.train_config
    save_config = global_config.save_config
    model_config = global_config.model_config
    img = Image.open(train_config.image_path).convert('RGB')
    img_np = np.array(img) / 255.0
    h, w, _ = img_np.shape
    logger.info(f'传入参数形状:(h,w)={(h,w)}')
    coords = get_coordinate_grid(h, w, torch.device('cpu'))
    coords = positional_encoding(coords)
    model = ConfigurableINRModel(model_config.config,in_features=coords.shape[-1])
    device = global_device
    model_path = os.path.join(save_config.model_save_path,save_config.model_name)
    model.load_state_dict(torch.load(model_path.__str__(),map_location=device))
    decompress_and_save(inr_model=model, model_input=coords, base_output_path=save_config.base_output_path, config=global_config)

