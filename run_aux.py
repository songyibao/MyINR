import argparse

from src.utils.mlflow_exp import run_experiments, run_experiments_aux

if __name__ == "__main__":
    # 获取脚本参数,格式为: python run_exp.py --config_file "EXP"
    # 获取 config_file 参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", type=str, required=True)
    # args = parser.parse_args()
    run_experiments_aux(['auxmodel.toml'])
