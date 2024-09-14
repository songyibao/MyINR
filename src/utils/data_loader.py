# 初始化数据集和数据加载器
# dataset = ImageDataset(config['data_path'])
# dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
# 实现上面的ImageDataset

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.configs.config import GlobalConfig
from src.models.inputs import positional_encoding


def get_coords(h, w):
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(0, 1, h, dtype=torch.float32),
        torch.linspace(0, 1, w, dtype=torch.float32), indexing='ij'
    )
    return torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)


class ImageCompressionDataset(Dataset):
    def __init__(self, image_path):
        """
        初始化自定义数据集。

        参数:
        - image_path (str): 图像文件的路径。
        """
        # 加载彩色图像并转换为 RGB 模式
        self.img = Image.open(image_path).convert('RGB')
        self.img_tensor = ToTensor()(self.img)  # 转换为 PyTorch 张量，形状为 (3, H, W)

        # 获取图像的宽、高信息
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]

        self.coords = get_coords(self.h,self.w)  # 转换为 (h * w, 2)

        # 获取图像的像素值，形状为 (h * w, 3)
        self.pixels = self.img_tensor.permute(1, 2, 0).view(-1, 3)  # 转换为 (h * w, 3)

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return 1  # 只有一张图像

    def __getitem__(self, idx):
        """
        根据索引获取样本。

        参数:
        - idx (int): 索引值

        返回:
        - 坐标网格（形状为 (h * w, 2)）
        - 图像像素值（形状为 (h * w, 3)）
        """
        config = GlobalConfig()
        # 判断是否有 num_frequencies 这个key
        if config.model_config.num_frequencies is None:
            return self.coords, self.pixels, self.h, self.w
        else:
            return positional_encoding(self.coords,num_frequencies=GlobalConfig().model_config.num_frequencies), self.pixels, self.h, self.w

