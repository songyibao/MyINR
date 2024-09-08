import toml
import platform
import os

class ModelConfig:
    def __init__(self,project_root_path:str,model_config:dict):
        self.config = model_config
        self.num_frequencies = self.config['num_frequencies']
        # self.hidden_features = self.config['hidden_features']
        # self.hidden_layers = self.config['hidden_layers']
        # self.in_features = self.config['in_features']
        # self.out_features = self.config['out_features']

class TrainConfig:
    def __init__(self,project_root_path:str,train_config:dict):
        self.config = train_config
        self.image_path = os.path.join(project_root_path,self.config['image_path'])
        self.learning_rate = self.config['learning_rate']
        self.num_epochs = self.config['num_epochs']
        self.patience = self.config['patience']
        self.scheduler_step_size = self.config['scheduler_step_size']
        self.scheduler_gamma = self.config['scheduler_gamma']
        self.target_loss = self.config['target_loss']
        self.loss_type = self.config['loss_type']
class SaveConfig:
    def __init__(self,project_root_path:str,save_config:dict):
        self.config = save_config
        # self.base_output_path = self.config['base_output_path']
        # self.image_save_path = self.config['image_save_path']
        self.model_save_path = os.path.join(project_root_path,self.config['model_save_path'])
        self.base_output_path = os.path.join(project_root_path,self.config['base_output_path'])
        self.image_save_path = os.path.join(project_root_path,self.config['image_save_path'])
        self.model_name = self.config['model_name']
        if not os.path.exists(self.base_output_path):
            os.makedirs(self.base_output_path)
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)


class MiscConfig:
    def __init__(self,project_root_path:str,misc_config:dict):
        self.config = misc_config
        self.log_save_path = self.config['log_save_path']
class GlobalConfig:
    def __init__(self):
        # 获取当前文件的绝对路径
        self.current_file_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(self.current_file_path),'config.toml')
        self.config = toml.load(config_path)
        self.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(self.current_file_path)))
        self.train_config = TrainConfig(self.project_root_path,self.config['train'])
        self.model_config = ModelConfig(self.project_root_path,self.config['model'])
        self.save_config = SaveConfig(self.project_root_path,self.config['save'])
        self.misc_config = MiscConfig(self.project_root_path,self.config['misc'])
    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return str(self.config)