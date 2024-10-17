from src.utils.mlflow_exp import run_experiments

if __name__ == "__main__":
    # 示例：你可以在这里传入配置文件名的列表
    config_files = [
        # "EXP",  # 示例文件名
        # "LSIREN",
        # "SIREN",
        # "LearnableEmbedding",
        # "DINER",
        # "Gauss_Block",
        # "Gauss",
        # "WIRE_Block",
        # "WIRE",
        # "SIREN_Block",
        # "SIREN",
        "PE_MLP_Block",
        "PE_MLP",
        # "MLP_Block",
        # "MLP",
        # "LSIREN_Block",
        # "LSIREN",
        # "FINER",
        # "FINER_Block"
    ]

    # 运行实验
    run_experiments(config_files)
