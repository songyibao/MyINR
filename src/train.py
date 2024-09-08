import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.models.loss import LossRegistry
from src.configs.config import TrainConfig
from tqdm import tqdm

from src.utils.device import global_device
from src.utils.evaluate import evaluate_tensor, evaluate_tensor_h_w_3
from src.utils.log import logger

def train_inr(model_input, target_image, model,  config: TrainConfig,device=global_device,):
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    patience = config.patience

    best_val_loss = np.inf  # 初始化最好的验证损失为无穷大
    max_patience_counter = 0
    patience_counter = 0  # 耐心计数器
    # 将模型移动到合适的设备上
    logger.info(f"运行设备：{device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 添加学习率调度器
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # VGG视觉感知损失
    loss_class = LossRegistry.get(config.loss_type)
    criterion = loss_class()
    # 直接使用均方误差损失
    # criterion = nn.MSELoss()

    last_res = None
    with tqdm(total=num_epochs, desc=f"Training:") as pbar:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # 获取模型输出
            output = model(model_input)

            # 将模型输出 reshape 成图像的尺寸
            output_image = output.view(target_image.shape)

            # 计算 psnr
            evaluate_res = evaluate_tensor_h_w_3(target_image,torch.clamp(output_image,0,1))
            # 计算损失
            loss = criterion(output_image, target_image)
            loss.backward()



            optimizer.step()

            # 学习率调度器步进
            scheduler.step()


            # 早停逻辑
            val_loss = loss.item()
            if val_loss<config.target_loss:
                tqdm.write(f"当前损失{val_loss}小于目标损失{config.target_loss}，停止训练。")
                break
            if patience_counter > max_patience_counter:
                max_patience_counter = patience_counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # 重置耐心计数器
            else:
                patience_counter += 1  # 增加耐心计数器

            if patience_counter >= patience:
                tqdm.write(f"早停: 在epoch {epoch + 1}停止训练。验证损失没有在{patience}个epoch内改善。")
                break

            update_value = {
                "Epoch": f'{epoch + 1:<{len(str(num_epochs))}}/{num_epochs}',
                "LR": f'{scheduler.get_last_lr()[0]:.6f}',
                "Loss": f'{loss.item():.4f}',
                "Patience": f'{patience_counter:<{len(str(patience))}}/{patience}',
                "Best Loss": f'{best_val_loss:.4f}',
                "Max Patience": f'{max_patience_counter:>4}'
            }
            update_value.update(evaluate_res)
            last_res = update_value
            # Update progress bar
            pbar.set_postfix(update_value)
            pbar.update()

    logger.info(f'训练完成,最后结果{last_res}')
    return model
