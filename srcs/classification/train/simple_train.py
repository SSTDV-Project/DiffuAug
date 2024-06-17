import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

from DiffuAug.srcs.datasets.duke_for_classification import DukeDatasetClassification
from DiffuAug.srcs import utility
from DiffuAug.srcs.classification.models.simple_cnn import SimpleCNN


def train():
    epochs = 50
    lr = 0.0001
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    
    # model = SimpleCNN().to(device)
    model = resnet18()    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features # 마지막 계층의 입력 특징의 수를 가져옴
    model.fc = nn.Linear(num_ftrs, 1)  # 마지막 계층을 새로운 클래스 수에 맞게 교체 (여기서는 10개 클래스)
    model = model.to(device)

    # optimizer, loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    # Load metadata
    classification_csv_root_path = r"/workspace/DiffuAug/metadata/classification/csv/0.8_0.1_0.1_balanced"
    train_csv_path = os.path.join(classification_csv_root_path, 'train_dataset.csv')
    val_csv_path = os.path.join(classification_csv_root_path, 'val_dataset.csv')
    test_csv_path = os.path.join(classification_csv_root_path, 'test_dataset.csv')
    
    # 데이터 셋 선언
    train_dataset = DukeDatasetClassification(
        csv_path=train_csv_path,
    )
    val_dataset = DukeDatasetClassification(
        csv_path=val_csv_path,
    )
    test_dataset = DukeDatasetClassification(
        csv_path=test_csv_path,
    )
    
    # 데이터 로더 선언
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Train
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, data in enumerate(train_loader):
            inputs, targets, _ = data
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

            logit = model(inputs)
            prob = torch.sigmoid(logit)
            loss = loss_func(prob, targets)

            # 값이 0인 텐서를 만든 후, 임계값을 기준으로 값을 1로 설정
            threshold = prob > 0.5
            predicted = torch.zeros_like(prob)
            predicted[threshold] = 1.0

            # backpropagation을 위해 gradient를 0으로 설정합니다.
            optimizer.zero_grad()
            loss.backward()
            
            # optimization 수행
            optimizer.step()
            epoch_loss += loss.item()
        
        # validation acc 계산
        total = 0
        correct = 0
        for idx, data in enumerate(val_loader):
            inputs, targets, _ = data
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

            with torch.no_grad():
                logit = model(inputs)
            prob = torch.sigmoid(logit)
            threshold = prob > 0.5
            predicted = torch.zeros_like(prob)
            predicted[threshold] = 1.0

            total += predicted.size(0)
            correct += (predicted == targets).sum().item()
            
            log_targets = targets.squeeze(-1)
            log_predicted = predicted.squeeze(-1)
            print("Targets: ", log_targets)
            print("Predicted: ", log_predicted)
            
        acc = 100.0 * (correct / total)
        print("Validation ACC: ", acc)


if __name__ == "__main__":
    train()