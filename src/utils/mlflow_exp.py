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
from src.models.model import ConfigurableINRModel, ConfigurableBlockModelNew, AuxModel
from src.train import train_inr, train_inr_aux
from src.utils.data_loader import ImageCompressionDataset, get_coords, ImgDataset2dBlock, reconstruct_tensor, \
    ImgDataset1dBlock
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
    model_class = ConfigurableINRModel
    inr_model = model_class(config.net, in_features=coords.shape[-1], out_features=c)

    summary(inr_model, input_data=coords.to('cpu'), depth=10)  # show all layers

    # 训练模型
    trained_inr_model,best_output = train_inr(model_input=coords, target=original_pixels, model=inr_model, device=device,
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
    logger.info(f'测试完整图像的重建结果')
    output_h_w_c = best_output.cpu().reshape(h, w, c)
    eval_res = evaluate_tensor_h_w_3(dataset.img, output_h_w_c)
    logger.info(f'{eval_res}')
    if mlflow.active_run() is not None:
        mlflow.log_param("final_PSNR",eval_res["PSNR"])
    logger.info("转换和保存图像")
    reconstruct_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())
    original_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())

    # 创建并保存实验摘要
    exp_summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Config": config.model_dump(exclude_none=True),
        "Evaluation Results": eval_res
    }
    s_str = str(summary(inr_model.cpu(), input_data=coords.cpu(),verbose=0))
    exp_summary.update({"Model Summary": s_str})

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
    # decompress_and_save(inr_model=model, base_output_path=config.save.base_output_path,
    #                     config=config, model_input=coords, original_image=original_image)
def exp_block_1d(config: MyConfig, device: torch.device):
    # 本实验针对config.net.use_block_model = True且config.net.h_blocks = None 且config.net.w_blocks = None
    assert config.net.use_block_model and config.net.h_blocks is None and config.net.w_blocks is None
    logger.info(f'模型配置:{config.net.model_dump(exclude_none=True)}')
    logger.info("加载数据")
    dataset = ImgDataset1dBlock(config)
    logger.info("数据加载完成")
    model_class = ConfigurableBlockModelNew
    output_list = []
    H, W, C = dataset.h, dataset.w, dataset.channels
    model_summary_list = []
    for i in range(dataset.__len__()):
        coords_block, target_block = dataset[i]
        logger.info(f'input:{coords_block.shape}, target:{target_block.shape}')

        inr_model = model_class(config.net, in_features=coords_block.shape[-1], out_features=target_block.shape[-1])
        logger.info(f"正在拟合一维展平均分块blocks[{i}]")
        model_summary_list.append(summary(inr_model, input_data=coords_block.to('cpu'), depth=10,verbose=0))
        _, output = train_inr(model_input=coords_block, target=target_block, model=inr_model, device=device,
                              train_config=config.train,block_index=i)
        output_list.append(output)

    output = torch.cat(output_list, dim=0)
    output = torch.clamp(output, 0, 1)
    output_h_w_c = output.cpu().reshape(H, W, C)
    original_img = dataset.img.cpu()
    logger.info(f'测试完整图像的重建结果')
    eval_res = evaluate_tensor_h_w_3(original_img, output_h_w_c)
    logger.info(f'{eval_res}')
    if mlflow.active_run() is not None:
        mlflow.log_param("final_PSNR",eval_res["PSNR"])
    logger.info("转换和保存图像")
    reconstruct_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())
    original_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())

    # 创建并保存实验摘要
    exp_summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Config": config.model_dump(exclude_none=True),
        "Evaluation Results": eval_res,
        "Model Summary": model_summary_list
    }

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
def exp_block_2d(config: MyConfig, device: torch.device):
    # 本实验针对config.net.use_block_model = True且config.net.h_blocks not None 且config.net.w_blocks not None
    assert config.net.use_block_model and config.net.h_blocks is not None and config.net.w_blocks is not None
    logger.info(f'模型配置:{config.net.model_dump(exclude_none=True)}')
    logger.info("加载数据")
    dataset = ImgDataset2dBlock(config)
    logger.info("数据加载完成")
    model_class = ConfigurableBlockModelNew
    output_list = []
    H, W, C = dataset.h, dataset.w, dataset.channels
    positions = dataset.positions # [{"pos":[x,y], "size":[h,w]},{},{}]
    model_summary_list = []
    for i in range(dataset.__len__()):
        coords_block, target_block = dataset[i]
        logger.info(f'input:{coords_block.shape}, target:{target_block.shape}')

        inr_model = model_class(config.net, in_features=coords_block.shape[-1], out_features=target_block.shape[-1])
        logger.info(f"正在拟合二维空间相邻块blocks[{i}]")
        model_summary_list.append(summary(inr_model, input_data=coords_block.to('cpu'), depth=10,verbose=0))
        _, output = train_inr(model_input=coords_block, target=target_block, model=inr_model, device=device,
                              train_config=config.train,block_index=i)
        output_image = output.cpu().reshape(positions[i]['size'])
        output_list.append(output_image)

    output_h_w_c = reconstruct_tensor(output_list, positions , original_shape=(H, W, C))
    output_h_w_c = torch.clamp(output_h_w_c, 0, 1)
    logger.info(f'测试完整图像的重建结果')
    eval_res = evaluate_tensor_h_w_3(dataset.pixels, output_h_w_c)
    logger.info(f'{eval_res}')
    if mlflow.active_run() is not None:
        mlflow.log_param("final_PSNR",eval_res["PSNR"])
    logger.info("转换和保存图像")
    reconstruct_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())
    original_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().numpy())

    # 创建并保存实验摘要
    exp_summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Config": config.model_dump(exclude_none=True),
        "Evaluation Results": eval_res,
        "Model Summary": model_summary_list
    }

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
            # exp_block_2d(config=config,device=best_device)  # 执行实验
            if config.net.use_block_model:
                if config.net.h_blocks is None and config.net.w_blocks is None:
                    exp_block_1d(config=config, device=best_device)
                elif config.net.h_blocks is not None and config.net.w_blocks is not None:
                    exp_block_2d(config=config, device=best_device)
                else:
                    raise ValueError("Invalid block configuration")
            else:
                exp(config=config, device=best_device)
            logger.info("====================END===============================")

    logger.info("All experiments completed successfully!!!.")

def exp_aux(config: MyConfig, device: torch.device):
    logger.info(f'模型配置:{config.net.model_dump(exclude_none=True)}')
    logger.info("加载和预处理图像")
    dataset = ImageCompressionDataset(config)
    logger.info(f"创建坐标网格(包含位置编码)")
    coords, original_pixels, h, w, c = dataset[0]
    logger.info(f'{coords.shape}')
    inr_model = AuxModel(in_features=coords.shape[-1], out_features=c,mlp_width=128,mlp_depth=3,coords_shape=coords.shape)
    # inr_model = AuxModel(net_config=config.net,in_features=coords.shape[-1], out_features=c)
    summary(inr_model, input_data=coords.to('cpu'), depth=10)  # show all layers

    # 训练模型
    trained_inr_model = train_inr_aux(model_input=coords, target=original_pixels, model=inr_model, device=device,
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
    logger.info(f'测试完整图像的重建结果')
    trained_inr_model.phase = 2
    output_h_w_c = trained_inr_model(coords)['y'].cpu().reshape(h, w, c)
    eval_res = evaluate_tensor_h_w_3(dataset.img, output_h_w_c)
    logger.info(f'{eval_res}')
    if mlflow.active_run() is not None:
        mlflow.log_param("final_PSNR",eval_res["PSNR"])
    logger.info("转换和保存图像")
    reconstruct_image_mlfow_obj = mlflow.Image(output_h_w_c.cpu().detach().numpy())
    original_image_mlfow_obj = mlflow.Image(dataset.img.cpu().detach().numpy())

    # 创建并保存实验摘要
    exp_summary = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Config": config.model_dump(exclude_none=True),
        "Evaluation Results": eval_res
    }
    s_str = str(summary(inr_model.cpu(), input_data=coords.cpu(),verbose=0))
    exp_summary.update({"Model Summary": s_str})

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
def run_experiments_aux(config_files):
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
            # exp_block_2d(config=config,device=best_device)  # 执行实验
            exp_aux(config=config, device=best_device)
            logger.info("====================END===============================")

    logger.info("All experiments completed successfully!!!.")
