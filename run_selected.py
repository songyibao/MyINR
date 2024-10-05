import os
import mlflow
import skimage
import torch
from src.configs.config import MyConfig
from src.utils.log import logger
from tests.encode import exp

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

        # 开始MLflow运行并进行实验
        with mlflow.start_run() as run:
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

if __name__ == "__main__":
    # 示例：你可以在这里传入配置文件名的列表
    config_files = [
        "SIREN",  # 示例文件名
        "LSIREN"
    ]

    # 运行实验
    run_experiments(config_files)
