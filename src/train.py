import copy

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.configs.config import TrainConfig
from src.models.loss import LossRegistry
from src.utils.device import global_device
from src.utils.evaluate import evaluate_tensor_h_w_3
from src.utils.log import logger



def train_inr(model_input, target_image, model, train_config: TrainConfig, device=global_device):
    learning_rate = train_config.learning_rate
    num_steps = train_config.num_steps
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
    scheduler = StepLR(optimizer, step_size=train_config.scheduler_step_size, gamma=train_config.scheduler_gamma)

    loss_class = LossRegistry.get(train_config.loss_type)
    criterion = loss_class()

    with tqdm(total=num_steps, desc=f"Training:") as pbar:
        for epoch in range(num_steps):
            optimizer.zero_grad()

            output = model(model_input)
            output_image = output.view(target_image.shape)

            loss = criterion(output_image, target_image)
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
                best_model_state = model.state_dict()
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

            evaluate_res = evaluate_tensor_h_w_3(target_image, torch.clamp(output_image, 0, 1))
            update_value.update(evaluate_res)
            pbar.set_postfix(update_value)
            pbar.update()

    logger.info(f'模型训练完成,测试图像重建结果')
    model.load_state_dict(best_model_state)
    output = model(model_input)
    output_image = output.view(target_image.shape)
    evaluate_res = evaluate_tensor_h_w_3(target_image, torch.clamp(output_image, 0, 1))
    logger.info(evaluate_res)

    return model
