import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def compute_optimal_threshold(y_true, y_scores):
    """
    최적의 임계값을 찾아 sensitivity와 specificity 계산
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic을 계산하여 최적의 임계값 찾기
    j_scores = tpr - fpr
    optimal_threshold_index = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_threshold_index]
    sensitivity = tpr[optimal_threshold_index]
    specificity = 1 - fpr[optimal_threshold_index]
    
    return optimal_threshold, sensitivity, specificity, roc_auc


def compute_auc_with_slices(prediction_csv_path):
    y_true, y_scores = get_true_and_scores_with_slices(prediction_csv_path)

    auc_score = roc_auc_score(y_true, y_scores)
    print("AUC: {:.2f}".format(auc_score))
    
    # AUC, Sensitivity, Specificity 계산
    # _, sensitivity, specificity, roc_auc = compute_optimal_threshold(y_true, y_scores)
    # print(f"AUC: {roc_auc:.2f}")
    # print(f"Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}\n")


def draw_roc_curve(pred_result_csv_path, save_curve_png_path):
    """
    Test set에 대한 예측 결과를 읽어서 ROC Curve를 그립니다.
    """
    # ROC Curve 그리기를 위한 준비
    plt.figure()
    if not os.path.exists(save_curve_png_path):
        os.makedirs(save_curve_png_path)
    save_curve_png_path = os.path.join(save_curve_png_path, "roc_curve.png")
    
    # CSV에서 True와 예측 Score 추출
    y_true, y_scores = get_true_and_scores_with_slices(pred_result_csv_path)
    
    # ROC Curve 그림
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 각 fold의 ROC Curve 추가
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    
    # ROC Curve 그래프 설정
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Cross Validation')
    plt.legend(loc="lower right")
    plt.savefig(save_curve_png_path)


def compute_acc_with_slices(prediction_csv_path):
    """
    CSV 파일을 불러와서 ACC를 계산합니다.

    Args:
        prediction_csv_root_path (str): Test fold에 대한 예측 결과가 저장된 CSV 파일의 경로입니다.
    """
    print("------------------------------------------")
    
    df = pd.read_csv(prediction_csv_path)
    
    correct_slice = df[df['Predicted'] == df['Targets']]
    acc = 100.0 * len(correct_slice) / len(df)
    print(f"Accuracy: {acc:.2f}%\n")


def get_true_and_scores_with_slices(pred_result_csv_path):
    """
    CSV에 작성된 예측 결과를 읽어서 실제 label값 True와 예측 Score(확률)를 반환합니다.
    """
    # 예측 결과 CSV 파일 읽기
    pred_result_df = pd.read_csv(pred_result_csv_path)
    
    # True와 예측 Score 추출
    y_true = pred_result_df['Targets'].values
    y_scores = pred_result_df['Probs'].values
    
    # numpy의 1차원 배열로 변환
    y_true = np.array(y_true).ravel()
    y_scores = np.array(y_scores).ravel()
    
    return y_true, y_scores

