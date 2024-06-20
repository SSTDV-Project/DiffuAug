import logging

import numpy as np

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, RandomSampler, Subset

from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

from DiffuAug.srcs.datasets.duke_for_generation import DukeDataset


def example_fid1():
    """
    FID metric을 사용하는 계산 예시
    """
    _ = torch.manual_seed(123)

    fid = FrechetInceptionDistance(feature=64)

    # 약간 겹치는 image intensity 분포 두 개 생성
    imgs_dist1 = torch.randint(0, 200, (2, 3, 10, 10), dtype=torch.uint8) # 3채널인 10x10 이미지 2개
    imgs_dist2 = torch.randint(100, 255, (2, 3, 10, 10), dtype=torch.uint8) # 3채널인 10x10 이미지 2개
    
    print("imgs_dist1 shape: ", imgs_dist1.shape)
    print("imgs_dist2 shape: ", imgs_dist2.shape)
    
    fid.update(imgs_dist1, real=True) # 진짜 분포
    fid.update(imgs_dist2, real=False) # 가짜 분포
    fid_result = fid.compute()
    
    print(fid_result)


def example_fid2():
    """
    1채널 이미지를 대상으로 FID metric을 사용하는 계산 예시. FID 계산을 위해 
    같은 이미지를 3채널로 변환하여 사용한다.
    """
    _ = torch.manual_seed(123)

    fid = FrechetInceptionDistance(feature=64)

    # 약간 겹치는 image intensity 분포 두 개 생성
    imgs_dist1 = torch.randint(0, 200, (2, 1, 10, 10), dtype=torch.uint8)
    imgs_dist2 = torch.randint(100, 255, (2, 1, 10, 10), dtype=torch.uint8)
    
    imgs_dist1_3chan = imgs_dist1.repeat(1, 3, 1, 1)
    imgs_dist2_3chan = imgs_dist2.repeat(1, 3, 1, 1)
    
    fid.update(imgs_dist1_3chan, real=True)
    fid.update(imgs_dist2_3chan, real=False)
    fid_result = fid.compute()
    
    print(fid_result)
    

def compute_fid(device, original_data_path, generated_data_path):
    """
    여기서 사용하는 DukeDataset 클래스는 neg디렉토리와 pos디렉토리를 합쳐서 모든 종류에 대한 이미지를 평가한다.
    """
    IMG_SIZE = 299
    BATCH_SIZE = 100
        
    img_transform = transforms.Compose(
        [
            transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
                                                                                                                                      
    # 참고: https://github.com/mueller-franzes/medfusion/blob/main/scripts/evaluate_images.py
    fid = FrechetInceptionDistance(feature=64, normalize=True) 
    fid.to(device)
    
    # Dataset 선언
    dataset_real = DukeDataset(
        data_dir=original_data_path, 
        transform=img_transform,
        target_label="all",
        )
    dataset_fake = DukeDataset(
        data_dir=generated_data_path,
        transform=img_transform,
        target_label="all",
        )
    
    # Dataloader 선언
    dl_real = DataLoader(
        dataset=dataset_real,
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        drop_last=False
        )
    dl_fake = DataLoader(
        dataset=dataset_fake,
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        drop_last=False
        )

    # Start FID Calculation
    for (imgs_real_batch, label) in tqdm(dl_real):     
        imgs_real_batch = imgs_real_batch.repeat(1, 3, 1, 1)
        imgs_real_batch = imgs_real_batch.to(device)
        
        fid.update(imgs_real_batch, real=True)

    for (imgs_fake_batch, label) in tqdm(dl_fake):     
        imgs_fake_batch = imgs_fake_batch.repeat(1, 3, 1, 1)
        imgs_fake_batch = imgs_fake_batch.to(device)
        
        fid.update(imgs_fake_batch, real=False)

    # FID 계산 값 출력
    fid_result = fid.compute()
    print(f"FID Score: {fid_result:.2f}")    