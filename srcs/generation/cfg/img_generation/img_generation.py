import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from DiffuAug.srcs.generation.cfg.models.cfg_duke import (GaussianDiffusion, UnetModel)


def load_model(cfg, model_path):
    if model_path is not None:
        model = UnetModel(
            in_channels=cfg.params.channels,
            model_channels=cfg.params.init_channels,
            out_channels=cfg.params.channels,
            channel_mult=cfg.params.dim_mults,
            attention_resolutions=[],
            class_num=cfg.cfg_params.class_num
        )
    
        # 모델 로드    
        states = torch.load(model_path, map_location='cuda')
        model = model.to("cuda")
        model.load_state_dict(states['model_state'])

    return model

def generate_cond_ddpm_img(cfg, model_path):
    model = load_model(cfg, model_path)
    
    # 샘플링 폴더 생성
    sampling_root_path = cfg.paths.sampling_path
    Path(sampling_root_path).mkdir(parents=True, exist_ok=True)
    
    # Diffusion 연산 유틸 선언
    gaussian_diffusion = GaussianDiffusion(timesteps=cfg.params.timesteps, beta_schedule='linear')
 
    # 설정 값 로드
    img_size = cfg.params.img_size
    n_class = cfg.cfg_params.class_num
    
    # 한 번에 생성할 이미지 수
    batch_size = 16
    
    # 생성할 이미지의 class 선택
    select_class_num = 0
    clas_name = "pos" if select_class_num == 1 else "neg"
    print("class_name: ", clas_name)

    # Class별 DDPM 이미지 생성 
    target_imgs_num = 2605
    iteration_num = target_imgs_num // batch_size
    mod = target_imgs_num % batch_size
    # 나머지가 있는 경우 반복 횟수에 1 추가
    total_iteration_num = iteration_num if mod == 0 else iteration_num + 1
    print("total_iteration_num: ", total_iteration_num)

    for i in tqdm(range(total_iteration_num)):
        generated_images = gaussian_diffusion.sample(
            model=model,
            image_size=img_size,
            batch_size=batch_size, 
            channels=1, 
            n_class=n_class, 
            w=cfg.cfg_params.w, 
            mode='select',
            select_class_num=0,
            clip_denoised=False
            )
        
        imgs = generated_images[-1]
        print(imgs.shape)
        
        imgs = imgs.reshape(batch_size, img_size, img_size)
        for j in range(batch_size):
            img = imgs[j]
            img = img + 1.0
            img = img * 255 / 2
            
            img_save_path = os.path.join(sampling_root_path, f"{clas_name}_img_{i*batch_size+j}.png")
            cv2.imwrite(img_save_path, img)


def generate_cond_ddim_img(cfg, model_path):
    model = load_model(cfg, model_path)
    
    # 샘플링 폴더 생성
    sampling_root_path = cfg.paths.sampling_path
    Path(sampling_root_path).mkdir(parents=True, exist_ok=True)
    
    # Diffusion 연산 유틸 선언
    gaussian_diffusion = GaussianDiffusion(timesteps=cfg.params.timesteps, beta_schedule='linear')
    
    # 설정 값 로드
    img_size = cfg.params.img_size
    n_class = cfg.cfg_params.class_num
    
    # 한 번에 생성할 이미지 수
    batch_size = cfg.generation.gen_img_num
    
    # 생성할 이미지의 class 선택
    select_class_num = cfg.cfg_params.sampling_class_num
    clas_name = "pos" if select_class_num == 1 else "neg"
    print("class_name: ", clas_name)
    
    # 반환된 값의 이미지, 레이블 인덱스
    IMG_IDX = 0
    LABEL_IDX = 1

    # Class별 DDIM 이미지 생성 
    target_imgs_num = 2000
    iteration_num = target_imgs_num // batch_size
    mod = target_imgs_num % batch_size
    # 나머지가 있는 경우 반복 횟수에 1 추가
    total_iteration_num = iteration_num if mod == 0 else iteration_num + 1
    print("total_iteration_num: ", total_iteration_num)
    
    for i in range(total_iteration_num):    
        generated_images = gaussian_diffusion.ddim_sample(
            model=model, 
            image_size=img_size, 
            batch_size=batch_size, # 한 번에 생성할 이미지 수
            channels=1,
            ddim_timesteps=100,
            n_class=n_class,
            w=cfg.cfg_params.w,
            mode='select',
            ddim_discr_method='quad',
            ddim_eta=0.0,
            clip_denoised=False,
            select_class_num=select_class_num
            )
        
        imgs = generated_images[-1][IMG_IDX]
        labels = generated_images[-1][LABEL_IDX]
        
        print(imgs.shape)
        
        imgs = imgs.reshape(batch_size, img_size, img_size)
        for j in range(batch_size):
            img = imgs[j]
            img = img + 1.0
            img = img * 255 / 2
            
            img_save_path = os.path.join(sampling_root_path, f"ddim_{clas_name}_img_{i*batch_size+j}.png")
            cv2.imwrite(img_save_path, img)