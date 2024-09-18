
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from piq import SSIMLoss
# 定义一个装饰器，用于注册损失函数,损失函数的输入形状统一为 [height, width, 3]
class LossRegistry:
    _loss_functions = {}

    @classmethod
    def register(cls, name):
        def decorator(loss_function):
            cls._loss_functions[name] = loss_function
            return loss_function
        return decorator

    @classmethod
    def get(cls, name):
        return cls._loss_functions.get(name)

@LossRegistry.register("VGGPerceptualLoss")
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # 使用预训练的 VGG19 模型
        self.vgg = models.vgg19(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结 VGG19 的权重

    def forward(self, x, y):
        # 确保输入是四维张量 [batch_size, channels, height, width]
        if x.dim() != 4 or y.dim() != 4:
            # print(f"输入必须是四维张量 [batch_size, channels, height, width], 但是输入的维度是: {x.shape}, y={x.shape}")
            if x.shape[-1]==3:
                # print(f"最后一维度是3，尝试调整维度")
                x = x.permute(2,0,1).unsqueeze(0)
                y = y.permute(2,0,1).unsqueeze(0)



        # 归一化到 VGG19 预期的输入范围
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = F.mse_loss(x_vgg, y_vgg)
        return loss


@LossRegistry.register("MixedLoss")
class MixedLoss(nn.Module):
    def __init__(self, perceptual_loss_weight=0.1):
        super(MixedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss_weight = perceptual_loss_weight

    def forward(self, output_image, target_image):
        perceptual_loss_value = self.perceptual_loss(output_image, target_image)
        mse_loss_value = self.mse_loss(output_image, target_image)
        total_loss = mse_loss_value + self.perceptual_loss_weight * perceptual_loss_value
        return total_loss


@LossRegistry.register("EdgeLoss")
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def rgb_to_grayscale(self,image):
        # 假设 image 是 [B, 3, H, W]，其中 B 是 batch size，3 是 RGB 通道
        r, g, b = image[:, 0:1, :, :], image[:, 1:2, :, :], image[:, 2:3, :, :]
        grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b
        return grayscale_image
    def edge_loss(self,output_image, target_image):
        is_rgb = False
        if output_image.shape[-1] == 3:
            is_rgb = True
        output_image = output_image.permute(2, 0, 1).unsqueeze(0)
        target_image = target_image.permute(2, 0, 1).unsqueeze(0)
        if is_rgb:
            output_image = self.rgb_to_grayscale(output_image)
            target_image = self.rgb_to_grayscale(target_image)
        # 使用 Sobel 算子来计算图像的梯度（边缘信息）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                               device=output_image.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                               device=output_image.device).unsqueeze(0).unsqueeze(0)

        output_grad_x = F.conv2d(output_image, sobel_x)
        output_grad_y = F.conv2d(output_image, sobel_y)
        target_grad_x = F.conv2d(target_image, sobel_x)
        target_grad_y = F.conv2d(target_image, sobel_y)

        output_grad = torch.sqrt(output_grad_x ** 2 + output_grad_y ** 2)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)

        return F.mse_loss(output_grad, target_grad)
    def forward(self, output_image, target_image):
        return self.edge_loss(output_image, target_image)





@LossRegistry.register("SSIMLoss")
class MySSIMLoss(nn.Module):
    def __init__(self):
        super(MySSIMLoss, self).__init__()
        self.ssim_loss = SSIMLoss(data_range=1.)
    def forward(self, output_image, target_image):
        output_image = output_image.permute(2, 0, 1).unsqueeze(0)
        target_image = target_image.permute(2, 0, 1).unsqueeze(0)
        output_image = torch.clamp(output_image, 0, 1)
        target_image = torch.clamp(target_image, 0, 1)
        return self.ssim_loss(output_image, target_image)

@LossRegistry.register("MSELoss")
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self, output_image, target_image):
        return self.mse_loss(output_image, target_image)

@LossRegistry.register("SSIMandEdgeLoss")
class SSIMandEdgeLoss(nn.Module):
    def __init__(self, edge_loss_weight=0.05, ssim_loss_weight=0.1):
        super(SSIMandEdgeLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        self.edge_loss_weight = edge_loss_weight
        self.ssim_loss_weight = ssim_loss_weight

    def forward(self, output_image, target_image):
        edge_loss_value = self.edge_loss(output_image, target_image)
        ssim_loss_value = self.ssim_loss(output_image, target_image)
        total_loss = self.edge_loss_weight * edge_loss_value + self.ssim_loss_weight * ssim_loss_value
        return total_loss



@LossRegistry.register("FullCombinedLoss")
class FullCombinedLoss(nn.Module):
    def __init__(self, perceptual_loss_weight=0.1, edge_loss_weight=0.05, ssim_loss_weight=0.1):
        super(FullCombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        self.perceptual_loss_weight = perceptual_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.ssim_loss_weight = ssim_loss_weight

    def forward(self, output_image, target_image):
        self.perceptual_loss.to(output_image.device)
        perceptual_loss_value = self.perceptual_loss(output_image, target_image)
        mse_loss_value = self.mse_loss(output_image, target_image)
        edge_loss_value = self.edge_loss(output_image, target_image)
        ssim_loss_value = self.ssim_loss(output_image, target_image)

        total_loss = (
            mse_loss_value +
            self.perceptual_loss_weight * perceptual_loss_value +
            self.edge_loss_weight * edge_loss_value +
            self.ssim_loss_weight * ssim_loss_value
        )
        return total_loss
