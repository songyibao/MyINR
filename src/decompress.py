import datetime
import os
import platform

import mlflow
import toml
import torch
from torchinfo import summary

from src.utils.data_loader import ImageCompressionDataset
from src.utils.device import global_device

os_type = platform.system()
if os_type == 'Windows':
    pass
from torchvision.transforms.functional import to_pil_image

from src.models.model1 import ConfigurableINRModel
from src.configs.config import MyConfig
from src.utils.evaluate import calculate_bpp, evaluate_tensor_h_w_3
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
            f.write(f"{key}: \n{value}\n")
def experiment_summary_to_text(summary: dict):
    """将实验摘要转换为文本"""
    text = ""
    for key, value in summary.items():
        text += f"{key}: \n{value}\n"
    return text

def create_comparison_image(original_image, reconstructed_image, save_path: str):
    """创建原始图像和重建图像的比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    original_image = original_image.cpu()
    # reconstructed_image.to('cpu')
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(reconstructed_image)
    ax2.set_title("Reconstructed Image")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def decompress_and_save(inr_model, config: MyConfig, base_output_path: str, model_input:torch.Tensor=None, original_image:torch.Tensor=None):
    # 创建本次实验目录
    experiment_dir = create_experiment_directory(base_output_path)

    # 使用CPU模式并设置模型为评估模式
    inr_model = inr_model.eval()
    device = next(inr_model.parameters()).device
    dataset = ImageCompressionDataset(config)
    dataset.img.save(os.path.join(experiment_dir, 'original_image.png'))
    if model_input is None or original_image is None:
        coords, original_image, h, w,c = dataset[0]
        original_image = original_image.view(h, w, c)
        model_input = coords.to(device)
        original_image = original_image.to(device)
    else:
        h,w,c = original_image.shape[0],original_image.shape[1],original_image.shape[2]
    logger.info(f"原图像形状{original_image.shape}")
    # 计算模型输出
    logger.info("计算模型输出")
    with torch.no_grad():
        output_image = inr_model(model_input).view(h, w, c) # [h*w,c]->[h,w,c]

    # 计算评估指标
    logger.info("计算原图像和重建图像psnr和ssim")

    with torch.no_grad():
        result=evaluate_tensor_h_w_3(original_image, torch.clamp(output_image, 0, 1))
    # 计算bpp
    logger.info("计算 bpp")
    bpp = calculate_bpp(original_image,inr_model)
    result.update({
        "bpp": bpp
    })
    logger.info(f'{result}')
    # 转换并保存图像
    logger.info("转换和保存图像")
    output_image = torch.clamp(output_image, 0, 1) # (h, w, c)
    reconstruct_image_mlfow_obj = mlflow.Image(output_image.cpu().numpy())
    original_image_mlfow_obj = mlflow.Image(original_image.cpu().numpy())

    # reconstructed_image = to_pil_image(output_image.permute(2, 0, 1)) # Tensor 类型会被内部 permute
    # img_save_path = os.path.join(experiment_dir, 'reconstructed_image.png')
    # reconstructed_image.save(img_save_path)

    # 创建比较图像
    # comparison_image_path = os.path.join(experiment_dir, 'comparison.png')
    # create_comparison_image(original_image, reconstructed_image, comparison_image_path)

    # 保存评估指标
    # result_file_path = os.path.join(experiment_dir, 'evaluation_results.toml')
    # save_config_to_toml(result, result_file_path)
    # mlflow.log_dict(result, "evaluation_results")

    # 保存配置文件
    # config_file_path = os.path.join(experiment_dir, 'config.toml')
    # save_config_to_toml(config.model_dump(), config_file_path)

    # 创建并保存实验摘要
    exp_summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Config": config.model_dump(),
        "Evaluation Results": result,
        # "Original Image": original_image_path,
        # "Reconstructed Image": img_save_path,
        # "Comparison Image": comparison_image_path,
    }
    s_str = str(summary(inr_model, input_data=model_input.to(device),verbose=0))
    exp_summary.update({"Model Summary": s_str})

    if mlflow.active_run() is not None:
        mlflow.log_image(reconstruct_image_mlfow_obj, "reconstructed_image.png")
        mlflow.log_image(original_image_mlfow_obj, "original_image.png")
        mlflow.log_text(experiment_summary_to_text(exp_summary), "experiment_summary.txt")
    else:
        logger.info("未找到活动的mlflow run, 无法记录实验结果到mlflow")
        reconstruct_image_mlfow_obj.save(os.path.join(experiment_dir, 'reconstructed_image.png'))
        original_image_mlfow_obj.save(os.path.join(experiment_dir, 'original_image.png'))
        save_experiment_summary(exp_summary, os.path.join(experiment_dir, 'experiment_summary.txt'))
        logger.info(f'实验结果已保存到目录: {experiment_dir}')

    return exp_summary





# 使用示例
# compress('data/raw/image.jpg', 'models/inr_model.pth', 'compressed_image.pth')
# Load and preprocess the image
if os.getenv('MODE',"TRAIN").upper()=="SINGLE":
    config = MyConfig.get_instance()
    dataset = ImageCompressionDataset(config)
    coords, original_pixels, h, w, c = dataset[0]
    model = ConfigurableINRModel(config.net, in_features=coords.shape[-1],out_features=c)
    model_path = os.path.join(config.save.net_save_path,config.save.net_name)
    model.load_state_dict(torch.load(model_path.__str__(),weights_only=True,map_location=torch.device('cpu')))
    decompress_and_save(inr_model=model, base_output_path=config.save.base_output_path, config=config)

