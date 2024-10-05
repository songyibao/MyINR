import mlflow
from src.configs.config import MyConfig
from src.utils.log import logger
from tests.encode import exp


# 加载配置
config = MyConfig.get_instance(config_name="SIREN", force_reload=True)


# 设置MLflow实验
mlflow.set_experiment(config.experiment_name)

# 开始MLflow运行并进行实验
with mlflow.start_run() as run:
    # print(f"Running experiment with config: {config_file}")
    logger.info(f"Running experiment with config: {config.experiment_name}")
    exp(config=config)  # 执行实验