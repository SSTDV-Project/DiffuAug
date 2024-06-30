import torch
import torch.nn as nn
# from torchview import draw_graph

from DiffuAug.srcs.generation.cfg.models.cfg_duke import train, UnetModel, generate_img
from DiffuAug.srcs import utility

YAML_PATH = r"/workspace/DiffuAug/exp_settings/configs/cfg/t500/full_data/cfg_64x64.yaml"

def vis_model(cfg):
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            batch_size = x.size(0)
            timesteps = torch.randint(0, cfg.params.timesteps, (batch_size,), dtype=torch.long).to(x.device)
            c = torch.randint(0, cfg.params.class_num, (batch_size,), dtype=torch.long).to(x.device)
            mask = torch.zeros(batch_size).int().to(x.device)
            
            return self.model(x, timesteps, c, mask)
    
    batch_size = cfg.params.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnetModel(
        in_channels=cfg.params.channels,
        model_channels=cfg.params.init_channels,
        out_channels=cfg.params.channels,
        channel_mult=cfg.params.dim_mults,
        attention_resolutions=[],
        class_num=cfg.cfg_params.class_num
    )
    model.to(device)
    
    wrapped_model = ModelWrapper(model)
    model_graph = draw_graph(
        wrapped_model, 
        input_size=(batch_size, cfg.params.channels, cfg.params.img_size, cfg.params.img_size), 
        device=device, 
        save_graph=True
        )


def main():    
    utility.set_seed()
    
    cfg = utility.load_config(YAML_PATH)
    cfg = utility.dict2namespace(cfg)
    utility.set_logger(cfg)
    
    # 학습
    train(cfg)


if __name__ == "__main__":
    main()