import os
import random

import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms


class DukeDatasetClassification(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transforms = transform
                
        df = pd.read_csv(self.csv_path)
        
        self.input_list = []
        self.label_list = []

        for index, row in df.iterrows():
            self.input_list.append(row['data_path'])
            self.label_list.append(row['label'])

    def __len__(self):
        if len(self.input_list) == len(self.label_list):
            return len(self.input_list)
        else:
            return "error"


    def standardization_norm(self, input_img):
        # epsilon을 더해주지 않으면, 0으로 값을 나누려는 경우가 생겨 device-side assert triggered가 발생할 수 있음.        
        epsilon = 1e-10
        mean_intensity = np.mean(input_img) # 의료 이미지의 평균 intensity 값을 계산        
        std_intensity = np.std(input_img) # 의료 이미지의 표준편차 intensity 값을 계산
        input_img = (input_img-mean_intensity) / (std_intensity+epsilon)
        # input_img /= 10
        return input_img 
    
    
    def min_max_normalization(self, input_img):
        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)
        return input_img

    
    def run_albumentation(self, input_img):
        # random.seed(990912)
        transform = A.Compose([
            A.Resize(64, 64),
            # A.HorizontalFlip(),
            # A.VerticalFlip(),
            A.Rotate(limit=(-15, 15), p=0.5),
            # A.ColorJitter(brightness=0,contrast=(1,5),saturation=0,hue=0), # Contrast
            # A.RandomBrightnessContrast(),
            # A.Sharpen(),
            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5), 
            ], is_check_shapes=False
        )
        augmentation = transform(image=input_img)
        return augmentation        


    def __getitem__(self, idx):        
        input_path = self.input_list[idx]
        label = self.label_list[idx]
        
        # 이미지 로드
        input_img = Image.open(input_path)
        input_img = np.array(input_img)
        
        # 어그멘테이션
        if self.transforms:
            # NOTE: Alubmentation을 통해 color jittering을 수행할 경우 input_img의 dtype이 float, unit8로 변환되어야 함.
            # Float로 변환되어도 소수점이 생길 뿐, 값 손실이 생기지 않으므로 괜찮음.
            input_img = input_img.astype(np.float32)
            augmentations = self.run_albumentation(input_img)
            input_img = augmentations['image']
        
        # 이미지 표준화
        input_img = self.standardization_norm(input_img)
        # print(f"input_img value: {input_img}")
        
        input_img = torch.from_numpy(input_img) # 우선 tensor타입으로 변환되는 것도 확인하였음. torch.float64로 변환됨.
        input_img = input_img.unsqueeze(0) # 채널 차원 추가

        if label == "pos":
            label = 1
        else:
            label = 0      

        return input_img, label, input_path