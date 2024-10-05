import os
import mlflow
import skimage
import torch
from src.configs.config import MyConfig
from src.utils.log import logger
from tests.encode import exp

# 设定要读取的配置文件夹路径
config_folder_path = "src/configs"  # 这里替换为你的文件夹路径

# 获取指定文件夹下所有的 .toml 文件
config_files = [f for f in os.listdir(config_folder_path) if f.endswith('.toml')]
# 移除 'config.toml' 文件
config_files = [f for f in config_files if f != 'config.toml']

# 设置默认的精度
torch.set_float32_matmul_precision('medium')

# 遍历每个配置文件并进行实验
for config_file in config_files:
    # 加载配置
    config = MyConfig.get_instance(config_name=config_file, force_reload=True)

    # 设置MLflow实验
    mlflow.set_experiment(config.experiment_name)
    # 开始MLflow运行并进行实验
    with mlflow.start_run() as run:
        # 记录该次实验使用的图片, 方便在 mlflow ui 中进行分类查看
        dataset = mlflow.data.from_numpy(features=skimage.io.imread(config.train.image_path), source=config.train.image_path,name=os.path.basename(config.train.image_path))
        mlflow.log_input(dataset)
        # print(f"Running experiment with config: {config_file}")
        logger.info("===================================================")
        logger.info(f"Running experiment with config: {config_file}, dataset: {config.train.image_path}")
        exp(config=config)  # 执行实验