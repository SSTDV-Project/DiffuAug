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


def load_model(device, mode=None):
    # Renset
    if mode=="resnet18":
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features # 마지막 계층의 입력 특징의 수를 가져옴
        model.fc = nn.Linear(num_ftrs, 1) 
        
        model = model.to(device=device)    
    # Alexnet
    elif mode=="alexnet":
        model = models.alexnet(pretrained=False)
        # 첫 번째 컨볼루션 레이어 수정 (1채널 입력)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

        # 분류기 수정 (2 클래스 분류 -> 1 출력 노드)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
        model = model.to(device=device)
    elif mode=="simple":
        model = SimpleCNN()
        model = model.to(device=device)
    
    return model


def train_one_epoch(cfg, model, optimizer, loss_func, train_loader, batch_size, epoch, device):
    """ 
    모델을 한 epoch 동안 훈련합니다.

    Args:
        model: 학습시킬 모델
        optimizer: 사용할 optimizer
        loss_func: 사용할 loss 함수
        trainloader: 학습 데이터를 담고 있는 DataLoader
        batch_size: 배치 사이즈
        epoch: 현재 epoch
        device: 텐서를 올릴 디바이스

    Returns:
        net: 학습된 모델
        epoch_loss: epoch의 평균 loss
    """
    print(f'Starting training: epoch {epoch}')
    model.train()
    
    y_scores_list = list()
    y_true_list = list()
    epoch_loss = 0.0    

    # 훈련 데이터에 대해 DataLoader를 반복합니다.
    for idx, data in enumerate(train_loader):
        # 입력을 가져오기
        inputs, targets, _ = data
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

        # forward pass 수행
        logit = model(inputs)        
        prob = torch.sigmoid(logit)
        
        loss = loss_func(prob, targets)

        # 값이 0인 텐서를 만든 후, 임계값을 기준으로 값을 1로 설정
        pivot = prob > cfg.params.threshold
        predicted = torch.zeros_like(prob)
        predicted[pivot] = 1.0

        # backpropagation을 위해 gradient를 0으로 설정합니다.
        optimizer.zero_grad()
        loss.backward()
        
        # optimization 수행
        optimizer.step()

        # loss값을 출력
        epoch_loss += loss.item()
        
        # ROC 계산을 위해 값 저장)
        y_scores_list.extend(prob.detach().cpu().numpy())
        y_true_list.extend(targets.cpu().numpy())
        
        if idx % batch_size == batch_size-1:
            batch_loss = loss.item()
            print(f"Loss after mini-batch {idx + 1}: {batch_loss:.3f}")


    # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
    y_scores = np.array(y_scores_list)
    y_true = np.array(y_true_list)

    # AUC 및 Epoch 평균 loss 계산
    auc_scroe = roc_auc_score(y_true, y_scores)
    epoch_loss = round(epoch_loss / len(train_loader), 3)
    
    print(f"AUC: {auc_scroe:.3f}")
    print(f"epoch mean loss: {epoch_loss} \n")
    
    return model, epoch_loss


def valid_one_epoch(model, val_loader, cur_epoch, device):
    model.eval()

    total = 0
    correct = 0
    
    # ROC 계산을 위한 리스트 초기화
    y_scores_list = list()
    y_true_list = list()
    
    print("y_scores_list: ", y_scores_list)
    print("y_true_list: ", y_true_list)
    
    # 테스트 데이터를 반복하며 예측값을 생성한다
    for batch_idx, data in enumerate(val_loader):
        # 입력을 가져오기
        inputs, targets, _ = data
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

        # 출력을 생성하기
        with torch.no_grad():
            logit = model(inputs)
        prob = torch.sigmoid(logit)

        # 값이 0인 텐서를 만든 후, 임계값을 기준으로 값을 1로 설정
        threshold = prob > 0.5
        predicted = torch.zeros_like(prob)
        predicted[threshold] = 1.0
        
        # 정확도 계산
        total += predicted.size(0)
        correct += (predicted == targets).sum().item()
        
        # ROC 계산을 위해 값 저장
        y_scores_list.extend(prob.detach().cpu().numpy())
        y_true_list.extend(targets.cpu().numpy())

    y_scores = np.array(y_scores_list)
    y_true = np.array(y_true_list)
    print("y_scores.shape: ", y_scores.shape)
    print("y_true.shape: ", y_true.shape)

    # accuracy 출력 
    acc = 100.0 * (correct / total)
    auc = roc_auc_score(y_true, y_scores)
    
    print(f"Valid epoch: {cur_epoch}, correct: {correct}, total: {total}, ACC: {acc:.2f}, AUC: {auc:.2f}\n")
    
    return acc, auc


def test_one_epoch(cfg, model, test_loader, epoch, device, is_save_csv=False):
    print(f'Epoch: {epoch}, Starting testing')
    model.eval()
    correct, total = 0, 0
    
    epoch_input_paths = list()
    epoch_logit = list()
    epoch_probs = list()
    epoch_predicteds = list()
    epoch_targets = list()
    
    # ROC 계산을 위한 리스트 초기화
    y_scores_list = list()
    y_true_list = list()
    
    # 테스트 데이터를 반복하며 예측값을 생성한다
    for batch_idx, data in enumerate(test_loader, 0):
        # 입력을 가져오기
        inputs, targets, input_paths = data
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

        # 출력을 생성하기
        with torch.no_grad():
            logit = model(inputs)
        prob = torch.sigmoid(logit)

        # 값이 0인 텐서를 만든 후, 임계값을 기준으로 값을 1로 설정
        threshold = prob > 0.5
        predicted = torch.zeros_like(prob)
        predicted[threshold] = 1.0
        
        # 정확도 계산
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # fold별 결과 저장
        epoch_input_paths.append(input_paths)
        epoch_logit.append(logit)
        epoch_probs.append(prob)
        epoch_predicteds.append(predicted)
        epoch_targets.append(targets)
        
        # ROC 계산을 위해 값 저장
        y_scores_list.extend(prob.detach().cpu().numpy())
        y_true_list.extend(targets.cpu().numpy())
        # y_scores_list.extend(predicted.cpu().numpy())
        # y_true_list.extend(targets.cpu().numpy())
    
    # 에포크별 예측값에 대한 결과를 CSV로 저장합니다.
    if is_save_csv:
        savecsv_prediction_results_for_epoch(
            input_paths=epoch_input_paths,
            logits=epoch_logit,
            probs=epoch_probs, 
            predicted=epoch_predicteds,
            targets=epoch_targets, 
            current_epoch=epoch,
            save_path=cfg.paths.test_predict_result_save_path
            )

    # accuracy 출력 
    acc = 100.0 * (correct / total)
    auc = roc_auc_score(y_true_list, y_scores_list)
    print(f"Test epoch: {epoch}, correct: {correct}, total: {total}, ACC: {acc:.2f}, AUC: {auc:.2f}\n")
    
    return acc, auc


def savecsv_prediction_results_for_epoch(
    input_paths, 
    logits,
    probs,
    predicted,
    targets,
    current_epoch, 
    save_path
    ):
    """
    Fold단위의 예측 결과를 CSV로 저장합니다.

    Args:
        input_paths (List[List[[str]]): 한 폴드의 파일 경로가 담긴 리스트. 리스트 내부 리스트는 각 미니배치의 파일 경로를 담고 있습니다.
        predicted (List[torch.Tensor]): 모델이 예측한 값
        targets (List[torch.Tensor]): 실제 label 값
        current_fold: 현재 fold 번호
        save_path: CSV를 저장할 경로
    """
    dataframe_cols = ["FIlePath", "Targets", "Predicted", "Probs", "Logit"]
    predicted_results = list()
    
    for row, file_paths in enumerate(input_paths):
        for col, file_path in enumerate(file_paths):
            predicted_results.append([ 
                                      file_path, 
                                      targets[row][col].item(), 
                                      predicted[row][col].item(),
                                      round(probs[row][col].item(), 3),
                                      round(logits[row][col].item(), 3)
                                      ]
                                     )
    
    # DataFrame으로 변환 후 CSV로 저장
    csv_save_path = f"{save_path}/predicted_{current_epoch}.csv"
    if not os.path.exists(os.path.dirname(csv_save_path)):
        os.makedirs(os.path.dirname(csv_save_path))
    
    df = pd.DataFrame(predicted_results, columns=dataframe_cols)
    df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')


def train(cfg):
    loss_and_auc_each_epoch = {}
    best_auc = 0.0
    
    csv_root_path = cfg.paths.csv_root_path
    train_csv_path = os.path.join(csv_root_path, "train_dataset.csv")
    val_csv_path = os.path.join(csv_root_path, "val_dataset.csv")
    test_csv_path = os.path.join(csv_root_path, "test_dataset.csv")
    
    # 데이터 셋 선언
    train_dataset = DukeDatasetClassification(
        csv_path=train_csv_path,
        transform=True
        )
    val_dataset = DukeDatasetClassification(
        csv_path=val_csv_path,
        transform=True
        )
    test_dataset = DukeDatasetClassification(
        csv_path=test_csv_path,
        transform=True
        )
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    print(f"test_dataset dataset length: {len(test_dataset)}")
    
    # 데이터 로더 선언
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.params.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.params.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.params.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 모델 선언
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = load_model(device, mode="resnet18")
    model = resnet18()    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features # 마지막 계층의 입력 특징의 수를 가져옴
    model.fc = nn.Linear(num_ftrs, 1)  # 마지막 계층을 새로운 클래스 수에 맞게 교체 (여기서는 10개 클래스)
    model = model.to(device)
    
    # 옵티마이저 선언
    optimizer = optim.Adam(model.parameters(), lr=cfg.params.lr)
    
    # loss 함수 선언
    loss_func = nn.BCELoss()
    
    #  --------------- 한 곳에서 trian, test, val 모두 확인 -----------------#
    # Train
    for epoch in range(50):
        train_epoch_loss = 0
        train_total = 0
        train_correct = 0
        y_train_scores_list = list()
        y_train_true_list = list()
        print("Epoch: ", epoch)
        
        for idx, data in enumerate(train_loader):
            # model.train()
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

            train_total += predicted.size(0)
            train_correct += (predicted == targets).sum().item()

            # backpropagation을 위해 gradient를 0으로 설정합니다.
            optimizer.zero_grad()
            loss.backward()
            
            # optimization 수행
            optimizer.step()
            train_epoch_loss += loss.item()
            
            y_train_scores_list.extend(prob.detach().cpu().numpy())
            y_train_true_list.extend(targets.cpu().numpy())

        train_epoch_mean_loss = round(train_epoch_loss / len(train_loader), 3)

        # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
        y_train_scores = np.array(y_train_scores_list)
        y_train_true = np.array(y_train_true_list)
        auc_train_score = roc_auc_score(y_train_true, y_train_scores)

        train_acc = 100.0 * (train_correct / train_total)
        print(f"Train_acc: {train_acc}, correct: {train_correct}, total: {train_total}")
        print(f"Train AUC: {auc_train_score}")
        print(f"Train Epoch mean loss: {train_epoch_mean_loss} \n")
        
        # validation acc 계산
        total = 0
        correct = 0
        y_scores_list = list()
        y_true_list = list()
        val_epoch_loss = 0.0
        for idx, data in enumerate(val_loader):
            model.eval()
            inputs, targets, _ = data
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

            with torch.no_grad():
                logit = model(inputs)
                prob = torch.sigmoid(logit)
                loss = loss_func(prob, targets)
            threshold = prob > 0.5
            predicted = torch.zeros_like(prob)
            predicted[threshold] = 1.0

            total += predicted.size(0)
            correct += (predicted == targets).sum().item()
            
            log_targets = targets.squeeze(-1)
            log_predicted = predicted.squeeze(-1)
            # print("Targets: ", log_targets)
            # print("Predicted: ", log_predicted)
            
            y_scores_list.extend(prob.detach().cpu().numpy())
            y_true_list.extend(targets.cpu().numpy())
            val_epoch_loss += loss.item()

        val_epoch_mean_loss = round(val_epoch_loss / len(val_loader), 3)
        
        # AUC 및 Epoch 평균 loss 계산
        # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
        y_scores = np.array(y_scores_list)
        y_true = np.array(y_true_list)        
        auc_scroe = roc_auc_score(y_true, y_scores)
            
        acc = 100.0 * (correct / total)
        print(f"Validation ACC: {acc}, correct: {correct}, total: {total}")
        print(f"Validation AUC: {auc_scroe}")
        print(f"Validation Epoch mean loss: {val_epoch_mean_loss} \n")
        
        # Test 성능 계산
        test_total = 0
        test_correct = 0
        y_test_scores_list = list()
        y_test_true_list = list()
        test_epoch_loss = 0.0
        for idx, data in enumerate(test_loader):
            model.eval()
            inputs, targets, _ = data
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

            with torch.no_grad():
                logit = model(inputs)
                prob = torch.sigmoid(logit)
                loss = loss_func(prob, targets)
            threshold = prob > 0.5
            predicted = torch.zeros_like(prob)
            predicted[threshold] = 1.0

            test_total += predicted.size(0)
            test_correct += (predicted == targets).sum().item()
            
            log_targets = targets.squeeze(-1)
            log_predicted = predicted.squeeze(-1)
            # print("Targets: ", log_targets)
            # print("Predicted: ", log_predicted)
            
            y_test_scores_list.extend(prob.detach().cpu().numpy())
            y_test_true_list.extend(targets.cpu().numpy())
            test_epoch_loss += loss.item()

        test_epoch_mean_loss = round(test_epoch_loss / len(val_loader), 3)
        
        # AUC 및 Epoch 평균 loss 계산
        # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
        y_scores = np.array(y_test_scores_list)
        y_true = np.array(y_test_true_list)        
        auc_scroe = roc_auc_score(y_true, y_scores)
            
        acc = 100.0 * (test_correct / test_total)
        print(f"Test ACC: {acc}, correct: {test_correct}, total: {test_total}")
        print(f"Test AUC: {auc_scroe}")
        print(f"Test Epoch mean loss: {test_epoch_mean_loss} \n")

    #  --------------- 한 곳에서 trian, test, val 모두 확인 -----------------#
    
    # for epoch in range(cfg.params.epochs):
    #     model, cur_loss = train_one_epoch(
    #         cfg=cfg,
    #         model=model,
    #         optimizer=optimizer,
    #         loss_func=loss_func,
    #         train_loader=train_loader,
    #         batch_size=cfg.params.batch_size,
    #         epoch=epoch,
    #         device=device
    #         )

    #     cur_acc, val_auc = valid_one_epoch(model, val_loader, epoch, device)
        
    #     # epoch별 acc, auc, training loss를 딕셔너리에 저장
    #     loss_and_auc_each_epoch[epoch] = {
    #         'accuracy': round(cur_acc, 3),
    #         'auc': round(val_auc, 3), 
    #         'loss': round(cur_loss, 3)
    #         }

    #     # best 모델 저장
    #     if val_auc > best_auc:
    #         if os.path.exists(cfg.paths.model_save_path):
    #             os.makedirs(cfg.paths.model_save_path, exist_ok=True)                
    #         best_auc = val_auc
    #         best_model_save_path = f"{cfg.paths.model_save_path}/model-{epoch}.pth"
                    
    #         utility.save_model(model, model.state_dict(), best_model_save_path)
    #         print('--------------------------------')
    #         print(f"Best Val AUC: {best_auc:.2f}, Current Val AUC: {val_auc:.2f}")
    #         print(f"!!best model saved!! epoch: {epoch}, Val ACC:{cur_acc:.2f}, Val AUC:{val_auc:.2f}")
    #         print('--------------------------------\n')

    #     test_acc, test_auc = test_one_epoch(
    #         cfg=cfg,
    #         model=model,
    #         test_loader=test_loader,
    #         epoch=epoch,
    #         device=device,
    #         is_save_csv=True
    #     )