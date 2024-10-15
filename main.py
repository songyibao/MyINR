import os

import mlflow
import numpy as np
import skimage

from src.configs.config import MyConfig
from src.utils.log import logger

from src.utils.mlflow_exp import exp

# 加载配置
config = MyConfig.get_instance(config_name="LSIREN", force_reload=True)

# 设置MLflow实验
mlflow.set_experiment(config.experiment_name)

# 开始MLflow运行并进行实验
with mlflow.start_run() as run:
    # 记录该次实验使用的图片, 方便在mlflow ui中进行分类查看
    dataset = mlflow.data.from_numpy(features=skimage.io.imread(config.train.image_path), source=config.train.image_path,name=os.path.basename(config.train.image_path))
    mlflow.log_input(dataset)

    # print(f"Running experiment with config: {config_file}")
    logger.info(f"Running experiment with config: {config.experiment_name}, dataset: {config.train.image_path}")
    exp(config=config)  # 执行实验