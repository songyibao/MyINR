import datetime
import os.path

import mlflow
import numpy as np
import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from torchinfo import summary
from src.configs.config import MyConfig
from src.decompress import decompress_and_save, experiment_summary_to_text, create_experiment_directory, \
    save_experiment_summary
from src.models.model import ConfigurableINRModel, ConfigurableStackedModel, ConfigurableBlockModel
from src.train import train_inr, train_inr_block
from src.utils.data_loader import ImageCompressionDataset, get_coords, ImgDatasetBlock, reconstruct_tensor
from src.utils.device import get_best_device
from src.utils.evaluate import evaluate_tensor_h_w_3
from src.utils.log import logger
import time
import skimage.io


def exp(config: MyConfig, device: torch.device):
    logger.info(f'模型配置:{config.net.model_dump(exclude_none=True)}')
    logger.info("加载和预处理图像")
    dataset = ImageCompressionDataset(config)
    logger.info(f"创建坐标网格(包含位置编码)")
    coords, original_pixels, h, w, c = dataset[0]
    logger.info(f'{coords.shape}')
    original_image = original_pixels.view(h, w, c)
    if config.net.use_block_model:
        model_class = ConfigurableBlockModel
        inr_model = model_class(config.net, in_features=coords.shape[-1], out_features=c, input_size=coords.shape[0])
    else:
        model_class = ConfigurableStackedModel if config.net.use_stack_model else ConfigurableINRModel
        inr_model = model_class(config.net, in_features=coords.shape[-1], out_features=c)

    summary(inr_model, input_data=coords.to('cpu'), depth=10)  # show all layers

    # 训练模型
    trained_inr_model = train_inr(model_input=coords, target_image=original_image, model=inr_model, device=device,
                                  train_config=config.train)

    if not os.path.exists(config.save.base_output_path):
        os.makedirs(config.save.base_output_path)
    trained_inr_model = trained_inr_model.to('cpu')
    torch.save(trained_inr_model.state_dict(), os.path.join(config.save.net_save_path, config.save.net_name).__str__())
    if mlflow.active_run() is not None:
        input_schema = Schema([TensorSpec(np.dtype(np.float32), coords.shape)])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (coords.shape[0], c))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pytorch.log_model(trained_inr_model, "model", signature=signature,
                                 pip_requirements=["torch", "-r requirements.txt"])
    # 保存模型
    logger.info("保存模型")

    logger.info("加载模型")
    if config.net.use_block_model:
        model_class = ConfigurableBlockModel
        model = model_class(config.net, in_features=coords.shape[-1], out_features=c, input_size=coords.shape[0])
    else:
        model_class = ConfigurableStackedModel if config.net.use_stack_model else ConfigurableINRModel
        model = model_class(config.net, in_features=coords.shape[-1], out_features=c)
    model.load_state_dict(
        torch.load(os.path.join(config.save.net_save_path, config.save.net_name).__str__(), weights_only=True,
                   map_location="cpu"))

    decompress_and_save(inr_model=model, base_output_path=config.save.base_output_path,
                        config=config, model_input=coords, original_image=original_image)
def exp_block(config: MyConfig, device: torch.device):
    logger.info(f'模型配置:{config.net.model_dump(exclude_none=True)}')
    logger.info("加载数据")
    dataset = ImgDatasetBlock(config)
    logger.info("数据加载完成")
    pixels = dataset.pixels
    model_class = ConfigurableBlockModel
    output_list = []
    H, W, C = dataset.h, dataset.w, dataset.channels
    positions = dataset.positions # [{"pos":[x,y], "size":[h,w]},{},{}]
    for i in range(dataset.__len__()):
        coords_block, target_block = dataset[i]
        logger.info(f'input:{coords_block.shape}, target:{target_block.shape}')
        inr_model = model_class(config.net, in_features=coords_block.shape[-1], out_features=target_block.shape[-1],input_size=pixels.shape[0])
        _, output = train_inr_block(model_input=coords_block, target_image=target_block, model=inr_model, device=device,
                              train_config=config.train)
        output_k_1 = output.cpu().reshape(positions[i]['size'])
        output_list.append(output_k_1)

    output_h_w_c = reconstruct_tensor(output_list, positions , original_shape=(H, W, C))
    output_h_w_c = torch.clamp(output_h_w_c, 0, 1)
    logger.info(f'测试完整图像的重建结果')
    eval_res = evaluate_tensor_h_w_3(dataset.pixels, output_h_w_c)
    logger.info("转换和保存图像")
    reconstruct_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())
    original_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())

    # 创建并保存实验摘要
    exp_summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Config": config.model_dump(exclude_none=True),
        "Evaluation Results": eval_res,
        # "Original Image": original_image_path,
        # "Reconstructed Image": img_save_path,
        # "Comparison Image": comparison_image_path,
    }
    # s_str = str(summary(inr_model, input_data=model_input.to(device),verbose=0))
    # exp_summary.update({"Model Summary": s_str})

    if mlflow.active_run() is not None:
        mlflow.log_image(reconstruct_image_mlfow_obj, "reconstructed_image.png")
        mlflow.log_image(original_image_mlfow_obj, "original_image.png")
        mlflow.log_text(experiment_summary_to_text(exp_summary), "experiment_summary.txt")
    else:
        experiment_dir = create_experiment_directory(config.save.base_output_path)
        logger.info("未找到活动的mlflow run, 无法记录实验结果到mlflow")
        reconstruct_image_mlfow_obj.save(os.path.join(experiment_dir, 'reconstructed_image.png'))
        original_image_mlfow_obj.save(os.path.join(experiment_dir, 'original_image.png'))
        save_experiment_summary(exp_summary, os.path.join(experiment_dir, 'experiment_summary.txt'))
        logger.info(f'实验结果已保存到目录: {experiment_dir}')

def run_experiments(config_files):
    """
    使用传入的配置文件列表运行实验。

    参数:
    config_files: 包含配置文件名的列表
    """
    # 设置默认的精度
    torch.set_float32_matmul_precision('medium')

    # 遍历每个配置文件并进行实验
    for config_file in config_files:
        # 加载配置
        config = MyConfig.get_instance(config_name=config_file, force_reload=True)
        # 设置MLflow实验
        mlflow.set_experiment(config_file)
        # run_name 加上 当前时间戳
        run_name = f"{config_file}_{int(time.time())}"
        best_device = get_best_device()
        # 开始MLflow运行并进行实验
        with mlflow.start_run(run_name=run_name) as run:
            # 使用的图片, 仅用于区分, 方便在 mlflow ui 中进行分类查看
            dataset = mlflow.data.from_numpy(
                features=skimage.io.imread(config.train.image_path),
                name=os.path.basename(config.train.image_path)
            )
            mlflow.log_input(dataset)

            logger.info("====================START===============================")
            logger.info(f"Running experiment with config: {config_file}, dataset: {config.train.image_path}")
            exp(config=config,device=best_device)  # 执行实验
            logger.info("====================END===============================")

    logger.info("All experiments completed successfully!!!.")
