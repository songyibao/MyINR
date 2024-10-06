import os.path

import mlflow
import torch
from torchinfo import summary
from src.configs.config import MyConfig
from src.decompress import decompress_and_save
from src.models.model import ConfigurableINRModel
from src.train import train_inr
from src.utils.data_loader import ImageCompressionDataset, get_coords
from src.utils.device import global_device
from src.utils.log import logger
import time
import skimage.io

def exp(config: MyConfig, device: torch.device=global_device):
    logger.info(f'模型配置:{config.net.model_dump(exclude_none=True)}')
    logger.info("加载和预处理图像")
    dataset = ImageCompressionDataset(config)
    logger.info(f"创建坐标网格(包含位置编码)")
    coords, original_pixels, h, w, c = dataset[0]
    logger.info(f'{coords.shape}')
    original_image = original_pixels.view(h, w, c)
    inr_model = ConfigurableINRModel(config.net, in_features=coords.shape[-1], out_features=c)
    summary(inr_model, input_data=coords.to('cpu'))

    # 训练模型
    trained_inr_model = train_inr(model_input=coords, target_image=original_image, model=inr_model, device=device,
                                  train_config=config.train)

    if not os.path.exists(config.save.base_output_path):
        os.makedirs(config.save.base_output_path)
    trained_inr_model = trained_inr_model.to('cpu')
    torch.save(trained_inr_model.state_dict(),os.path.join(config.save.net_save_path, config.save.net_name).__str__())
    if mlflow.active_run() is not None:
        mlflow.pytorch.log_model(trained_inr_model, "model")
    # 保存模型
    logger.info("保存模型")

    # 记录模型保存路径
    # logger.info("保存模型到wandb")
    logger.info("加载模型")
    model = ConfigurableINRModel(config.net, in_features=coords.shape[-1], out_features=c)
    # model.layers[0] = ConfigurableINRModel(config.pe_net, in_features=real_coords.shape[-1], out_features=learned_embedding.shape[-1])
    model.load_state_dict(
        torch.load(os.path.join(config.save.net_save_path, config.save.net_name).__str__(), weights_only=True,
                   map_location="cpu"))

    # summary(inr_model, input_data=real_coords.to(device))
    decompress_and_save(inr_model=model, base_output_path=config.save.base_output_path,
                        config=config,model_input=coords,original_image=original_image)
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
        mlflow.set_experiment(config.experiment_name)

        # run_name 设置为 config.experiment_name 加上 当前时间戳
        run_name = f"{config.experiment_name}_{int(time.time())}"
        # 开始MLflow运行并进行实验
        with mlflow.start_run(run_name = run_name) as run:
            # 记录该次实验使用的图片，方便在 mlflow ui 中进行分类查看
            dataset = mlflow.data.from_numpy(
                features=skimage.io.imread(config.train.image_path),
                source=config.train.image_path,
                name=os.path.basename(config.train.image_path)
            )
            mlflow.log_input(dataset)

            logger.info("===================================================")
            logger.info(f"Running experiment with config: {config_file}, dataset: {config.train.image_path}")
            exp(config=config)  # 执行实验