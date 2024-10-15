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

# def positional_encoding(coords, num_freqs=10):
#     """
#     对输入坐标进行位置编码。
#
#     参数:
#         coords (torch.Tensor): 输入坐标，形状为 (N, 2)，包含 N 个二维坐标。
#         num_freqs (int): 编码频率数量，控制编码的维度大小。
#
#     返回:
#         encoded_coords (torch.Tensor): 编码后的坐标，形状为 (N, 4 * num_freqs)。
#     """
#     # 生成频率序列
#     frequencies = torch.tensor([2 ** i for i in range(num_freqs)], dtype=torch.float32)
#
#     # 对 x 和 y 分别编码
#     x, y = coords[:, 0], coords[:, 1]
#     x_encoded = []
#     y_encoded = []
#
#     for freq in frequencies:
#         # 对 x 和 y 进行编码，并保留每个编码的维度
#         x_encoded.append(torch.sin(freq * x).unsqueeze(-1))  # (N, 1)
#         x_encoded.append(torch.cos(freq * x).unsqueeze(-1))  # (N, 1)
#         y_encoded.append(torch.sin(freq * y).unsqueeze(-1))  # (N, 1)
#         y_encoded.append(torch.cos(freq * y).unsqueeze(-1))  # (N, 1)
#
#     # 将 x 和 y 的编码结果分别拼接
#     x_encoded = torch.cat(x_encoded, dim=-1)  # 形状: (N, 2 * num_freqs)
#     y_encoded = torch.cat(y_encoded, dim=-1)  # 形状: (N, 2 * num_freqs)
#
#     # 合并 x 和 y 的编码结果，得到最终的编码表示
#     encoded_coords = torch.cat([x_encoded, y_encoded], dim=-1)  # 形状: (N, 4 * num_freqs)
#
#     return encoded_coords


def float_to_binary(tensor):
    # 将[0,1]之间的浮点数映射到[0,255]的整数
    scaled_tensor = (tensor * 255).int()

    # 将每个通道的像素值转换为二进制字符串，保持8位长
    binary_tensor = torch.cat([((scaled_tensor >> i) & 1).unsqueeze(-1) for i in range(7, -1, -1)], dim=-1)

    # 将结果转换为24位的形式（每个像素的3个通道展开为24个二进制位）
    return binary_tensor.view(-1, 24)

def find_sin_cos_intersections(points, k, frequency):
    N = points.shape[0]  # 点的数量
    intersections = torch.zeros((N, 4 * k))  # 初始化返回张量

    # 对于 y = sin(x) 和 y = a 的交点
    a_mask = (-1 <= points[:, 0]) & (points[:, 0] <= 1)  # 判断 a 是否在 [-1, 1] 范围内
    valid_a = points[a_mask, 0]  # 获取有效的 a 值
    if valid_a.shape[0] > 0:
        x_sol_sin = torch.arcsin(valid_a)  # arcsin 解
        sin_solutions_pos = x_sol_sin[:, None] + 2 * torch.pi * torch.arange(-k, k, device=points.device)[None, :]  # 正向解
        sin_solutions_neg = -sin_solutions_pos  # 对称解
        sin_solutions = torch.cat((sin_solutions_pos, sin_solutions_neg), dim=1)  # 合并正向和对称解
        sin_solutions = torch.sort(sin_solutions, dim=1)[0][:, :2 * k]  # 取前 2k 个解
        intersections[a_mask, :2 * k] = sin_solutions

    # 对于 y = cos(x) 和 y = b 的交点
    b_mask = (-1 <= points[:, 1]) & (points[:, 1] <= 1)  # 判断 b 是否在 [-1, 1] 范围内
    valid_b = points[b_mask, 1]  # 获取有效的 b 值
    if valid_b.shape[0] > 0:
        x_sol_cos = torch.arccos(valid_b)  # arccos 解
        cos_solutions_pos = x_sol_cos[:, None] + 2 * torch.pi * torch.arange(-k, k, device=points.device)[None, :]  # 正向解
        cos_solutions_neg = -cos_solutions_pos  # 对称解
        cos_solutions = torch.cat((cos_solutions_pos, cos_solutions_neg), dim=1)  # 合并正向和对称解
        cos_solutions = torch.sort(cos_solutions, dim=1)[0][:, :2 * k]  # 取前 2k 个解
        intersections[b_mask, 2 * k:] = cos_solutions

    # 通过另一个频率的三角函数映射到 [-1, 1] 范围
    mapped_intersections = torch.sin(frequency * intersections)  # 使用正弦函数进行映射

    return mapped_intersections


def cartesian_to_polar(coords, scale=1.0):
    """
    将笛卡尔坐标转换为极坐标，并将其与原直角坐标拼接，同时结果归一化到 [-1, 1]

    参数:
    - coords: [H*W, 2] 的张量，笛卡尔坐标系下的 (x, y) 坐标
    - scale: 控制转换后的 r 的缩放尺度，默认为 1.0

    返回:
    - [H*W, 4] 的张量，包含归一化的 (x, y, r, theta)
    """
    # 提取 x 和 y 坐标
    x = coords[:, 0]
    y = coords[:, 1]

    # 计算极径 r，并进行缩放
    r = torch.sqrt(x**2 + y**2) * scale

    # 计算极角 theta
    theta = torch.atan2(y, x)

    # 归一化 r
    r_max = r.max()
    r = (r / r_max) * 2 - 1  # 将 r 归一化到 [-1, 1]

    # 归一化 theta 到 [-1, 1]
    theta = theta / torch.pi  # 将 theta 归一化到 [-1, 1]

    # 将原始的 (x, y) 和计算出的 (r, theta) 拼接
    cartesian_polar_coords = torch.cat(( r.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)

    return cartesian_polar_coords


class FFM():
    def __init__(self, in_features: int, out_features: int, scale: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

    # 创建 B 矩阵
    def create_fixed_B(self, in_features, out_features, scale=10, seed=3407):
        torch.manual_seed(seed)
        B = torch.randn((in_features, out_features)) * scale
        torch.nn.init.uniform_(B, a=0, b=1)
        return B
        # 定义傅里叶特征映射函数

    def fourier_feature_mapping(self, x, B):
        x_proj = 60 * x @ B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def fmap(self, x):
        B = self.create_fixed_B(self.in_features, self.out_features, self.scale)
        return self.fourier_feature_mapping(x, B)
def get_coords(h, w):
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-2, 2, h, dtype=torch.float32),
        torch.linspace(-2, 2, w, dtype=torch.float32), indexing='ij'
    )
    res = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    return res


class ImageCompressionDataset(Dataset):
    def __init__(self, config: MyConfig, mode: str = 'test'):
        """
        初始化自定义数据集。

        参数:
        - image_path (str): 图像文件的路径。
        """
        self.config = config
        # 加载图像
        # self.img = Image.fromarray(skimage.data.camera()) # 读取示例图像
        self.img = Image.open(config.train.image_path)
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
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]
        self.coords = get_coords(self.h, self.w)  # 转换为 (h * w, 2)
        # self.coords = torch.linspace(-1, 1, self.h * self.w, dtype=torch.float32).reshape(-1, 1)
        # self.coords = cartesian_to_polar(self.coords)  # 转换为 (h * w, 4)
        # 获取图像的像素值，形状为 (h * w, 3)
        self.pixels = self.img_tensor.permute(1, 2, 0).view(-1, self.channels)



        if self.config.net.layers[0].type == 'LearnableEmbedding':
            self.coords = torch.arange(self.h * self.w).long()
        elif self.config.net.num_frequencies is not None:
            self.coords = positional_encoding(self.coords, num_frequencies=self.config.net.num_frequencies)
        elif self.config.net.degree is not None:
            self.coords = poly_fit(self.coords, degree=self.config.net.degree)
        elif self.config.net.ffm_out_features is not None:
            self.coords = FFM(in_features=self.coords.shape[-1], out_features=self.config.net.ffm_out_features).fmap(
                self.coords)
        elif self.config.net.use_polar_coords:
            self.coords = cartesian_to_polar(self.coords)

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return 1

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
