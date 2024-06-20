from DiffuAug.srcs.generation.cfg.img_generation.img_generation import (generate_cond_ddim_img, generate_cond_ddpm_img)
from DiffuAug.srcs import utility


def main():
    GEN_SETTING_YAML_PATH = r"/workspace/DiffuAug/srcs/generation/cfg/img_generation/configs/balanced/p_uncond_0.2/w_0.0-4.0_neg-pos.yaml"
    OPTION = 'ddim_class'
    
    utility.set_seed()
    
    cfg = utility.load_config(GEN_SETTING_YAML_PATH)
    cfg = utility.dict2namespace(cfg)
    
    if OPTION == 'ddim_class':
        # Class로 분간된 DDIM 이미지 생성
        generate_cond_ddim_img(cfg, cfg.paths.model_path)    
    elif OPTION == 'ddpm_class': 
        # class로 분간된 DDPM 이미지 생성
        generate_cond_ddpm_img(cfg, cfg.paths.model_path)    


if __name__ == '__main__':
    main()