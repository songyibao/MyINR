# 初始化数据集和数据加载器
# dataset = ImageDataset(config['data_path'])
# dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
# 实现上面的ImageDataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.configs.config import MyConfig
from src.models.inputs import positional_encoding
from sklearn.preprocessing import PolynomialFeatures


def upsample_image(img, scale_factor=2):
    h, w, c = img.shape
    upsampled_img = np.zeros((scale_factor * h, scale_factor * w, c), dtype=img.dtype)
    upsampled_img[::scale_factor, ::scale_factor, :] = img
    return upsampled_img


def downsample_image(img, scale_factor=2):
    h, w, c = img.shape
    downsampled_img = img[::scale_factor, ::scale_factor, :]
    return downsampled_img


def poly_fit(coords, degree=3):
    poly = PolynomialFeatures(degree=degree)
    return torch.tensor(poly.fit_transform(coords), dtype=torch.float32)


def get_coords(h, w, data_range: int = 1):
    x_coords, y_coords = None, None
    if data_range == 1:
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32), indexing='ij'
        )
    elif data_range == -1:
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(1, h, h, dtype=torch.float32),
            torch.linspace(1, w, w, dtype=torch.float32), indexing='ij'
        )
    else:
        raise ValueError(f"Unsupported data_range: {data_range}, must be 1 or -1")
    res = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    return res

class FFM():
    def __init__(self, in_features: int, out_features: int, scale: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

    # 创建 B 矩阵
    def create_fixed_B(self, in_features, out_features, scale=10, seed=3407):
        torch.manual_seed(seed)
        B = torch.randn((in_features, out_features)) * scale
        return B
        # 定义傅里叶特征映射函数

    def fourier_feature_mapping(self, x, B):
        x_proj = 2 * torch.pi * x @ B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def fmap(self, x):
        B = self.create_fixed_B(self.in_features, self.out_features, self.scale)
        return self.fourier_feature_mapping(x, B)


class ImageCompressionDataset(Dataset):
    def __init__(self, config: MyConfig, mode: str = 'test'):
        """
        初始化自定义数据集。

        参数:
        - image_path (str): 图像文件的路径。
        """
        self.config = config
        # 加载图像
        self.img = Image.fromarray(skimage.data.camera())
        # self.img = Image.open(config.train.image_path).convert(MyConfig.get_instance().mode)
        self.channels = -1
        # 判断图像的通道数
        if self.img.mode == 'RGB':
            self.channels = 3
        elif self.img.mode == 'L':
            self.channels = 1
        else:
            self.channels = len(self.img.getbands())
            if self.channels == 4:  # 如果是RGBA，转换为RGB
                self.img = self.img.convert('RGB')
                self.channels = 3
            else:
                raise ValueError(f"Unsupported image mode: {self.img.mode}")
        self.img_tensor = ToTensor()(self.img)  # 转换为 PyTorch 张量，形状为 (3, H, W)
        # 转换为 NumPy 数组
        # self.img_array = self.img_tensor.numpy().transpose(1, 2, 0)
        # 上采样
        # self.upsampled_img_array = upsample_image(self.img_array)
        # 转换回 PIL 图像（如果需要）
        # self.upsampled_img = Image.fromarray(self.upsampled_img_array)
        self.mode = mode
        # 保存或显示上采样后的图像
        # 获取图像的宽、高信息
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]

        self.coords = get_coords(self.h, self.w)  # 转换为 (h * w, 2)

        # 获取图像的像素值，形状为 (h * w, 3)
        self.pixels = self.img_tensor.permute(1, 2, 0).view(-1, self.channels)

        # self.h, self.w = self.upsampled_img_array.shape[:2]
        # self.coords = get_coords(self.h,self.w)
        # self.pixels = ToTensor()(self.upsampled_img).permute(1, 2, 0).view(-1, self.channels)

        if self.config.net.layers[0].type == 'LearnableEmbedding':
            self.coords = torch.arange(self.h * self.w).long()
        elif self.config.net.num_frequencies is not None:
            self.coords = positional_encoding(self.coords, num_frequencies=self.config.net.num_frequencies)
        elif self.config.net.degree is not None:
            self.coords = poly_fit(self.coords, degree=self.config.net.degree)
        elif self.config.net.ffm_out_features is not None:
            self.coords = FFM(in_features=self.coords.shape[-1], out_features=self.config.net.ffm_out_features).fmap(
                self.coords)

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        if self.mode == 'train':
            return self.config.train.num_steps
        elif self.mode == 'test':
            return 1
        else:
            raise ValueError(f"Invalid mode: {self.mode}, must be 'train' or 'test'")

    def __getitem__(self, idx):
        """
        根据索引获取样本。

        参数:
        - idx (int): 索引值

        返回:
        - 坐标网格（形状为 (h * w, 2)）
        - 图像像素值（形状为 (h * w, 3)）
        """
        return self.coords, self.pixels, self.h, self.w, self.channels
