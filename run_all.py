import os
from src.utils.mlflow_exp import run_experiments

if __name__ == "__main__":
    # 设定要读取的配置文件夹路径
    config_folder_path = "src/configs"  # 这里替换为你的文件夹路径

    # 获取指定文件夹下所有的 .toml 文件
    config_files = [f for f in os.listdir(config_folder_path) if f.endswith('.toml')]
    # 移除 'config.toml' 文件
    config_files = [f for f in config_files if f != 'config.toml']

    run_experiments(config_files=config_files)