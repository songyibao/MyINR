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

def float_to_binary(tensor):
    # 将[0,1]之间的浮点数映射到[0,255]的整数
    scaled_tensor = (tensor * 255).int()

    # 将每个通道的像素值转换为二进制字符串，保持8位长
    binary_tensor = torch.cat([((scaled_tensor >> i) & 1).unsqueeze(-1) for i in range(7, -1, -1)], dim=-1)

    # 将结果转换为24位的形式（每个像素的3个通道展开为24个二进制位）
    return binary_tensor.view(-1, 24)

def find_sin_cos_intersections(points, k):
    N = points.shape[0]  # 点的数量
    intersections = torch.zeros((N, 4 * k))  # 初始化返回张量

    for i in range(N):
        a, b = points[i]  # 获取点 (a, b)

        # 处理 y = a 和 y = sin(x) 的交点
        # 解方程 a = sin(x)，找到 x 的解
        sin_solutions = []
        if -1 <= a <= 1:  # arcsin 只有在 [-1, 1] 范围内才有解
            x_sol = torch.arcsin(a)  # 利用 torch 的 arcsin
            for n in range(-k, k):  # 关于 y 轴对称的 k 个解
                x_n = x_sol + 2 * torch.pi * n  # 利用 sin(x) 的周期性
                sin_solutions.append(x_n.item())
                sin_solutions.append(-x_n.item())  # 对称点

        # 取前 2k 个点
        sin_solutions = sorted(sin_solutions)[:2 * k]

        # 处理 y = b 和 y = cos(x) 的交点
        # 解方程 b = cos(x)，找到 x 的解
        cos_solutions = []
        if -1 <= b <= 1:  # arccos 只有在 [-1, 1] 范围内才有解
            x_sol = torch.arccos(b)  # 利用 torch 的 arccos
            for n in range(-k, k):  # 关于 y 轴对称的 k 个解
                x_n = x_sol + 2 * torch.pi * n  # 利用 cos(x) 的周期性
                cos_solutions.append(x_n.item())
                cos_solutions.append(-x_n.item())  # 对称点

        # 取前 2k 个点
        cos_solutions = sorted(cos_solutions)[:2 * k]

        # 将结果填入返回张量
        intersections[i, :2*k] = torch.tensor(sin_solutions)
        intersections[i, 2*k:] = torch.tensor(cos_solutions)

    return intersections

def cartesian_to_polar(coords, scale=2.0):
    """
    将笛卡尔坐标转换为极坐标，并将其与原直角坐标拼接
    参数:
    - coords: [H*W, 2] 的张量，笛卡尔坐标系下的 (x, y) 坐标
    - scale: 控制转换后的 r 的缩放尺度，默认为 1.0

    返回:
    - [H*W, 4] 的张量，包含 (x, y, r, theta)
    """
    # 提取 x 和 y 坐标
    x = coords[:, 0]
    y = coords[:, 1]

    # 计算极径 r，并进行缩放
    r = torch.sqrt(x**2 + y**2) * scale

    # 计算极角 theta
    theta = torch.atan2(y, x)

    # 将原始的 (x, y) 和计算出的 (r, theta) 拼接
    cartesian_polar_coords = torch.cat((coords, r.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)

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
        # self.img = Image.fromarray(skimage.data.camera()) # 读取示例图像
        self.img = Image.open(config.train.image_path).convert('L')
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
        # self.coords = find_sin_cos_intersections(self.coords, k=1)
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
