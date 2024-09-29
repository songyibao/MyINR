import os.path

import torch
from torchinfo import summary
from src.configs.config import MyConfig
from src.decompress import decompress_and_save
from src.models.model1 import ConfigurableINRModel
from src.train import train_inr, train_pe_inr
from src.utils.data_loader import ImageCompressionDataset, get_coords
from src.utils.device import global_device
from src.utils.log import logger


def test(config: MyConfig=MyConfig.get_instance(), device: torch.device=global_device):
    logger.info(f'模型配置:{config.net.model_dump()}')
    logger.info("加载和预处理图像")
    dataset = ImageCompressionDataset(config)
    logger.info(f"创建坐标网格(包含位置编码)")
    coords, original_pixels, h, w, c = dataset[0]
    logger.info(f'{coords.shape}')
    original_image = original_pixels.view(h, w, c)
    inr_model = ConfigurableINRModel(config.net, in_features=coords.shape[-1], out_features=c)
    summary(inr_model, input_data=coords.to('cpu'))

    torch.set_float32_matmul_precision('medium')
    # 训练模型
    trained_inr_model = train_inr(model_input=coords, target_image=original_image, model=inr_model, device=device,
                                  train_config=config.train)

    learned_embedding_layer = trained_inr_model.layers[0]


    learned_embedding = learned_embedding_layer(coords.to(device))
    # 清除learned_embedding的梯度
    learned_embedding = learned_embedding.detach()
    torch.save(learned_embedding, os.path.join(config.save.base_output_path, 'embedding.pth').__str__())
    # real_coords = get_coords(h, w,data_range=1)
    # pe_model = ConfigurableINRModel(config.pe_net, in_features=real_coords.shape[-1], out_features=learned_embedding.shape[-1])
    # trained_pe_model = train_pe_inr(model_input=real_coords, learned_embedding=learned_embedding, model=pe_model, device=device, train_config=config.train)
    # trained_inr_model.layers[0] = trained_pe_model

    if not os.path.exists(config.save.base_output_path):
        os.makedirs(config.save.base_output_path)
    torch.save(trained_inr_model.state_dict(),
               os.path.join(config.save.net_save_path, config.save.net_name).__str__())
    # 保存模型
    logger.info("保存模型")

    # 记录模型保存路径
    # logger.info("保存模型到wandb")
    logger.info("加载模型")
    model = ConfigurableINRModel(config.net, in_features=coords.shape[-1], out_features=c)
    # model.layers[0] = ConfigurableINRModel(config.pe_net, in_features=real_coords.shape[-1], out_features=learned_embedding.shape[-1])
    model.load_state_dict(
        torch.load(os.path.join(config.save.net_save_path, config.save.net_name).__str__(), weights_only=True,
                   map_location=device))
    # summary(inr_model, input_data=real_coords.to(device))
    decompress_and_save(inr_model=model, base_output_path=config.save.base_output_path,
                        config=config,model_input=coords,original_image=original_image)

    # 保存生成的图像到wandb
    # logger.info("保存生成的图像到wandb")


test()
