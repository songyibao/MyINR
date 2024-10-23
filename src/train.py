from copy import deepcopy

import mlflow
import numpy as np
import torch
import torch.optim as optim
from piq import TVLoss
from torch import nn
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torchinfo import summary
from tqdm import tqdm

from src.configs.config import TrainConfig
from src.models.loss import LossRegistry
from src.utils.evaluate import evaluate_tensor_h_w_3
from src.utils.log import logger
import torch.nn.functional as F



def train_inr(model_input, target_image, model, train_config: TrainConfig, device):
    """
    训练INR模型, 返回训练过程中最好的模型
    :param model_input: 输入模型的数据 (H*W, N)
    :param target_image: 目标图像 (H, W, C)
    :param model: 待训练的模型
    :param train_config: 训练配置
    :param device: 训练设备
    """
    learning_rate = train_config.learning_rate
    learning_rate_final_ratio = train_config.learning_rate_final_ratio
    num_steps = train_config.num_steps
    # num_steps = 10 # for test
    patience = train_config.patience

    best_val_loss = np.inf
    best_model_state = None
    max_patience_counter = 0
    patience_counter = 0

    logger.info(f"运行设备：{device}")
    model = model.to(device)
    model_input = model_input.to(device)
    target_image = target_image.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_final_ratio ** min(step / num_steps, 1))

    loss_class = LossRegistry.get(train_config.loss_type)
    criterion = loss_class()


    with tqdm(total=num_steps, desc=f"Training:") as pbar:
        for epoch in range(num_steps):
            optimizer.zero_grad()

            output = model(model_input)
            output_image = output.view(target_image.shape)

            loss = criterion(output_image, target_image)
            psnr = 10 * torch.log10(1 / loss)

            loss.backward()

            optimizer.step()
            scheduler.step()

            val_loss = loss.item()
            if val_loss < train_config.target_loss:
                tqdm.write(f"当前损失{val_loss}小于目标损失{train_config.target_loss}，停止训练。")
                break

            if patience_counter > max_patience_counter:
                max_patience_counter = patience_counter

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                tqdm.write(f"早停: 在epoch {epoch + 1}停止训练。验证损失没有在{patience}个epoch内改善。")
                break

            update_value = {
                "Epoch": f'{epoch + 1:<{len(str(num_steps))}}/{num_steps}',
                "LR": f'{scheduler.get_last_lr()[0]:.6f}',
                "Loss": f'{loss.item():.4f}',
                "Patience": f'{patience_counter:<{len(str(patience))}}/{patience}',
                "Best Loss": f'{best_val_loss:.4f}',
                "Max Patience": f'{max_patience_counter:>4}'
            }

            # evaluate_res = evaluate_tensor_h_w_3(target_image, torch.clamp(output_image, 0, 1)) # {'PSNR': float,'MS-SSIM': float}
            update_value.update({"PSNR": f'{psnr:.2f}'})
            pbar.set_postfix(update_value)
            pbar.update()
            if mlflow.active_run() is not None:
                mlflow.log_metrics({
                    "Loss": best_val_loss,
                    "PSNR": psnr.item(),
                    # "MS-SSIM": evaluate_res['MS-SSIM'],
                    "LR": scheduler.get_last_lr()[0],
                    "Patience": patience_counter
                }, step=epoch)


    logger.info(f'模型训练完成,测试图像重建结果')
    model.load_state_dict(best_model_state)
    output = model(model_input)
    output_image = output.view(target_image.shape)
    evaluate_res = evaluate_tensor_h_w_3(target_image, torch.clamp(output_image, 0, 1))
    logger.info(evaluate_res)
    # 用最好结果进行最后一次日志记录
    if mlflow.active_run() is not None:
        mlflow.log_metrics({
            "Loss": best_val_loss,
            "PSNR": evaluate_res['PSNR'],
            # "MS-SSIM": evaluate_res['MS-SSIM'],
            "LR": scheduler.get_last_lr()[0],
            "Patience": patience_counter
        }, step=num_steps)

    return model
def train_inr_block(model_input, target_image, model, train_config: TrainConfig, device):
    """
    训练INR模型, 返回训练过程中最好的模型
    :param model_input: 输入模型的数据 (H*W, N)
    :param target_image: 目标图像 (H, W, C)
    :param model: 待训练的模型
    :param train_config: 训练配置
    :param device: 训练设备
    """
    learning_rate = train_config.learning_rate
    learning_rate_final_ratio = train_config.learning_rate_final_ratio
    num_steps = train_config.num_steps
    # num_steps = 10 # for test
    patience = train_config.patience

    best_val_loss = np.inf
    best_model_state = None
    best_output = None
    max_patience_counter = 0
    patience_counter = 0

    logger.info(f"运行设备：{device}")
    model = model.to(device)
    model_input = model_input.to(device)
    target_image = target_image.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_final_ratio ** min(step / num_steps, 1))

    loss_class = LossRegistry.get(train_config.loss_type)
    criterion = loss_class()


    with tqdm(total=num_steps, desc=f"Training:") as pbar:
        for epoch in range(num_steps):
            optimizer.zero_grad()

            output = model(model_input)
            output_image = output.view(target_image.shape)

            loss = criterion(output_image, target_image)
            psnr = 10 * torch.log10(1 / loss)

            loss.backward()

            optimizer.step()
            scheduler.step()

            val_loss = loss.item()
            if val_loss < train_config.target_loss:
                tqdm.write(f"当前损失{val_loss}小于目标损失{train_config.target_loss}，停止训练。")
                break

            if patience_counter > max_patience_counter:
                max_patience_counter = patience_counter

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                best_output = output.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                tqdm.write(f"早停: 在epoch {epoch + 1}停止训练。验证损失没有在{patience}个epoch内改善。")
                break

            update_value = {
                "Epoch": f'{epoch + 1:<{len(str(num_steps))}}/{num_steps}',
                "LR": f'{scheduler.get_last_lr()[0]:.6f}',
                "Loss": f'{loss.item():.4f}',
                "Patience": f'{patience_counter:<{len(str(patience))}}/{patience}',
                "Best Loss": f'{best_val_loss:.4f}',
                "Max Patience": f'{max_patience_counter:>4}'
            }

            # evaluate_res = evaluate_tensor_h_w_3(target_image, torch.clamp(output_image, 0, 1)) # {'PSNR': float,'MS-SSIM': float}
            update_value.update({"PSNR": f'{psnr:.2f}'})
            pbar.set_postfix(update_value)
            pbar.update()
            if mlflow.active_run() is not None:
                mlflow.log_metrics({
                    "Loss": best_val_loss,
                    "PSNR": psnr.item(),
                    # "MS-SSIM": evaluate_res['MS-SSIM'],
                    "LR": scheduler.get_last_lr()[0],
                    "Patience": patience_counter
                }, step=epoch)


    logger.info(f'模型训练完成,测试图像重建结果')
    model.load_state_dict(best_model_state)
    # output = model(model_input)
    output_image = best_output.view(target_image.shape)
    # output_image = output.view(target_image.shape)
    evaluate_res = evaluate_tensor_h_w_3(target_image, torch.clamp(output_image, 0, 1))
    logger.info(evaluate_res)
    # 用最好结果进行最后一次日志记录
    if mlflow.active_run() is not None:
        mlflow.log_metrics({
            "Loss": best_val_loss,
            "PSNR": evaluate_res['PSNR'],
            # "MS-SSIM": evaluate_res['MS-SSIM'],
            "LR": scheduler.get_last_lr()[0],
            "Patience": patience_counter
        }, step=num_steps)

    return model,best_output
