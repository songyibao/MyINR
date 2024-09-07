import toml

# 创建配置字典
config = {
    "train_config": {
        'data_path': 'data/processed/',
        'batch_size': 32,
        'hidden_features': 256,
        'hidden_layers': 3,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'model_save_path': 'models/inr_model.pth'
    }
}

# 写入TOML文件
with open("test_config.toml", "w") as config_file:
    toml.dump(config, config_file)
