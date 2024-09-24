import os
from typing import List, Optional, ClassVar

import imageio.v3
import toml
from pydantic import BaseModel, field_validator, model_validator, PrivateAttr

# 定义项目根路径
current_file_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

class LayerConfig(BaseModel):
    type: str
    in_features: Optional[int] = None
    out_features: int

class NetConfig(BaseModel):
    num_frequencies: Optional[int] = None
    degree: Optional[int] = None
    layers: List[LayerConfig]
    in_features: Optional[int] = None
    ffm_out_features: Optional[int] = None


class TrainConfig(BaseModel):
    image_path: str
    learning_rate: float
    num_steps: int
    num_epochs: int
    patience: int
    scheduler_step_size: int
    scheduler_gamma: float
    target_loss: float
    loss_type: str
    h: Optional[int] = None
    w: Optional[int] = None

    @field_validator('image_path', mode='before')
    @classmethod
    def validate_image_path(cls, v):
        full_path = os.path.join(project_root_path, v)
        if not os.path.exists(full_path):
            raise ValueError(f"Image path does not exist: {full_path}")
        return full_path

    def model_post_init(self, __context):
        image_path = self.image_path
        image = imageio.v3.imread(image_path)
        self.h, self.w = image.shape[:2]

class SaveConfig(BaseModel):
    net_save_path: str
    net_name: str
    base_output_path: str
    image_save_path: str

    @field_validator('net_save_path','base_output_path','image_save_path', mode='before')
    @classmethod
    def validate_paths(cls, v):
        full_path = os.path.join(project_root_path, v)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

class MiscConfig(BaseModel):
    log_save_path: str

    @field_validator('log_save_path', mode='before')
    @classmethod
    def validate_log_path(cls, v):
        full_path = os.path.join(project_root_path, v)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

class MyConfig(BaseModel):
    mode: str
    train: TrainConfig
    save: SaveConfig
    misc: MiscConfig
    net: NetConfig
    pe_net: Optional[NetConfig] = None

    _instance: ClassVar[Optional['MyConfig']] = None  # 标记为类变量

    @field_validator('mode', mode='before')
    @classmethod
    def validate_mode(cls, v):
        if v not in ['RGB', 'L']:
            raise ValueError(f"Invalid image mode: {v}, must be 'RGB' or 'L'")
        return v

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            config_path = os.path.join(os.path.dirname(current_file_path), 'config.toml')
            # 加载 TOML 文件
            config_dict = toml.load(config_path)
            # 创建唯一实例
            cls._instance = cls(**config_dict)
        return cls._instance
