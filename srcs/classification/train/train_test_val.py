    #  --------------- 한 곳에서 trian, test, val 모두 확인 -----------------#
    # Train
    # for epoch in range(50):
    #     train_epoch_loss = 0
    #     train_total = 0
    #     train_correct = 0
    #     y_train_scores_list = list()
    #     y_train_true_list = list()
    #     print("Epoch: ", epoch)
        
    #     for idx, data in enumerate(train_loader):
    #         # model.train()
    #         inputs, targets, _ = data
    #         inputs = inputs.to(device=device, dtype=torch.float32)
    #         targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

    #         logit = model(inputs)
    #         prob = torch.sigmoid(logit)
    #         loss = loss_func(prob, targets)

    #         # 값이 0인 텐서를 만든 후, 임계값을 기준으로 값을 1로 설정
    #         threshold = prob > 0.5
    #         predicted = torch.zeros_like(prob)
    #         predicted[threshold] = 1.0

    #         train_total += predicted.size(0)
    #         train_correct += (predicted == targets).sum().item()

    #         # backpropagation을 위해 gradient를 0으로 설정합니다.
    #         optimizer.zero_grad()
    #         loss.backward()
            
    #         # optimization 수행
    #         optimizer.step()
    #         train_epoch_loss += loss.item()
            
    #         y_train_scores_list.extend(prob.detach().cpu().numpy())
    #         y_train_true_list.extend(targets.cpu().numpy())

    #     train_epoch_mean_loss = round(train_epoch_loss / len(train_loader), 3)

    #     # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
    #     y_train_scores = np.array(y_train_scores_list)
    #     y_train_true = np.array(y_train_true_list)
    #     auc_train_score = roc_auc_score(y_train_true, y_train_scores)

    #     train_acc = 100.0 * (train_correct / train_total)
    #     print(f"Train_acc: {train_acc}, correct: {train_correct}, total: {train_total}")
    #     print(f"Train AUC: {auc_train_score}")
    #     print(f"Train Epoch mean loss: {train_epoch_mean_loss} \n")
        
    #     # validation acc 계산
    #     total = 0
    #     correct = 0
    #     y_scores_list = list()
    #     y_true_list = list()
    #     val_epoch_loss = 0.0
    #     for idx, data in enumerate(val_loader):
    #         model.eval()
    #         inputs, targets, _ = data
    #         inputs = inputs.to(device=device, dtype=torch.float32)
    #         targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

    #         with torch.no_grad():
    #             logit = model(inputs)
    #             prob = torch.sigmoid(logit)
    #             loss = loss_func(prob, targets)
    #         threshold = prob > 0.5
    #         predicted = torch.zeros_like(prob)
    #         predicted[threshold] = 1.0

    #         total += predicted.size(0)
    #         correct += (predicted == targets).sum().item()
            
    #         log_targets = targets.squeeze(-1)
    #         log_predicted = predicted.squeeze(-1)
    #         # print("Targets: ", log_targets)
    #         # print("Predicted: ", log_predicted)
            
    #         y_scores_list.extend(prob.detach().cpu().numpy())
    #         y_true_list.extend(targets.cpu().numpy())
    #         val_epoch_loss += loss.item()

    #     val_epoch_mean_loss = round(val_epoch_loss / len(val_loader), 3)
        
    #     # AUC 및 Epoch 평균 loss 계산
    #     # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
    #     y_scores = np.array(y_scores_list)
    #     y_true = np.array(y_true_list)        
    #     auc_scroe = roc_auc_score(y_true, y_scores)
            
    #     acc = 100.0 * (correct / total)
    #     print(f"Validation ACC: {acc}, correct: {correct}, total: {total}")
    #     print(f"Validation AUC: {auc_scroe}")
    #     print(f"Validation Epoch mean loss: {val_epoch_mean_loss} \n")
        
    #     # Test 성능 계산
    #     test_total = 0
    #     test_correct = 0
    #     y_test_scores_list = list()
    #     y_test_true_list = list()
    #     test_epoch_loss = 0.0
    #     for idx, data in enumerate(test_loader):
    #         model.eval()
    #         inputs, targets, _ = data
    #         inputs = inputs.to(device=device, dtype=torch.float32)
    #         targets= targets.to(device=device, dtype=torch.float32).unsqueeze(-1)

    #         with torch.no_grad():
    #             logit = model(inputs)
    #             prob = torch.sigmoid(logit)
    #             loss = loss_func(prob, targets)
    #         threshold = prob > 0.5
    #         predicted = torch.zeros_like(prob)
    #         predicted[threshold] = 1.0

    #         test_total += predicted.size(0)
    #         test_correct += (predicted == targets).sum().item()
            
    #         log_targets = targets.squeeze(-1)
    #         log_predicted = predicted.squeeze(-1)
    #         # print("Targets: ", log_targets)
    #         # print("Predicted: ", log_predicted)
            
    #         y_test_scores_list.extend(prob.detach().cpu().numpy())
    #         y_test_true_list.extend(targets.cpu().numpy())
    #         test_epoch_loss += loss.item()

    #     test_epoch_mean_loss = round(test_epoch_loss / len(val_loader), 3)
        
    #     # AUC 및 Epoch 평균 loss 계산
    #     # 모든 배치에 대한 예측 확률과 실제 레이블을 하나의 배열로 합치기
    #     y_scores = np.array(y_test_scores_list)
    #     y_true = np.array(y_test_true_list)        
    #     auc_scroe = roc_auc_score(y_true, y_scores)
            
    #     acc = 100.0 * (test_correct / test_total)
    #     print(f"Test ACC: {acc}, correct: {test_correct}, total: {test_total}")
    #     print(f"Test AUC: {auc_scroe}")
    #     print(f"Test Epoch mean loss: {test_epoch_mean_loss} \n")

    #  --------------- 한 곳에서 trian, test, val 모두 확인 -----------------#