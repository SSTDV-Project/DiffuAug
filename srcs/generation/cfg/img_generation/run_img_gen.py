from DiffuAug.srcs.generation.cfg.img_generation.img_generation import (generate_cond_ddim_img, generate_cond_ddpm_img)
from DiffuAug.srcs import utility


def main():
    GEN_SETTING_YAML_PATH = r"/workspace/DiffuAug/srcs/generation/cfg/img_generation/configs/ddim/imbalanced/p_uncond_0.1/w_0.0-4.0_neg-pos.yaml"
    OPTION = 'ddim_class'
    
    utility.set_seed()
    
    cfg = utility.load_config(GEN_SETTING_YAML_PATH)
    cfg = utility.dict2namespace(cfg)

    if OPTION == 'ddim_class':
        # Class로 분간된 DDIM 이미지 생성
        # generate_cond_ddim_img(cfg, cfg.paths.model_path)

        w_list = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
            2.0, 3.0, 4.0
        ]

        # 지정된 w 값까지 이미지 생성
        for w in w_list:
            cfg.cfg_params.w = w
            print("cfg.cfg_params.w: ", cfg.cfg_params.w)

            for class_num in range(2):
                cfg.cfg_params.sampling_class_num = class_num
                print("cfg.cfg_params.sampling_class_num: ", cfg.cfg_params.sampling_class_num)
                           
                generate_cond_ddim_img(cfg, cfg.paths.model_path)   
            
    elif OPTION == 'ddpm_class': 
        # class로 분간된 DDPM 이미지 생성
        generate_cond_ddpm_img(cfg, cfg.paths.model_path)    


if __name__ == '__main__':
    main()