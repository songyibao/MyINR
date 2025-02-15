from src.utils.mlflow_exp import run_experiments

if __name__ == "__main__":
    # 示例：你可以在这里传入配置文件名的列表
    config_files = [
        # "EXP",  # 示例文件名
        # "LSIREN",
        # "SIREN",
        "LearnableEmbedding",
        # "DINER",
        # "SIREN",
        # "SIREN_Block_1d",
        # "SIREN_Block_2d",
        # "FINER",
        # "FINER_Block",
        # "Gauss_Block",
        # "Gauss",
        #
        # "WIRE_Block",
        # "WIRE",
        # "PE_MLP_Block",
        # "PE_MLP",
        # "MLP",
        # "MLP_Block_2d",
        # "MLP_Block_1d",

        # "LSIREN_Block",
        # "LSIREN",

    ]

    # 运行实验
    run_experiments(config_files)
