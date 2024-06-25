import os
import random

import pandas as pd
from natsort import natsorted
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


def gather_patient_data(root_dir, save_csv_path):
    # 데이터 저장을 위한 리스트
    data = []

    # 루트 디렉토리부터 하위 디렉토리 탐색
    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        
        # 환자 폴더가 디렉토리인지 확인
        if os.path.isdir(patient_path):
            for label in ['neg', 'pos']:
                label_path = os.path.join(patient_path, label)
                
                # label 폴더가 존재하는지 확인
                if os.path.isdir(label_path):
                    for file_name in os.listdir(label_path):
                        if file_name.endswith('.png'):
                            file_path = os.path.join(label_path, file_name)
                            data.append({
                                'patient_number': patient_folder,
                                'data_path': file_path,
                                'label': label
                            })

    # DataFrame 생성
    df = pd.DataFrame(data)

    # 자연 정렬
    df = df.reindex(natsorted(df.index, key=lambda x: (df.loc[x, 'patient_number'], df.loc[x, 'data_path'])))
    
    # CSV로 저장
    df.to_csv(os.path.join(save_csv_path, 'patient_data.csv'), index=False, encoding='utf-8-sig')


def balance_patient_data(input_csv_path, output_save_path):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)

    # 결과를 저장할 리스트
    balanced_data = []

    # 환자별로 그룹화
    grouped = df.groupby('patient_number')
    
    # 각 그룹에 대해 처리
    for patient_number, group in grouped:
        pos_data = group[group['label'] == 'pos']
        neg_data = group[group['label'] == 'neg']

        # pos 데이터의 개수를 기준으로 neg 데이터의 개수를 조절
        if len(neg_data) > len(pos_data):
            neg_data = neg_data.sample(len(pos_data), random_state=1)
        
        # 균형잡힌 데이터 추가
        balanced_group = pd.concat([pos_data, neg_data])
        balanced_data.append(balanced_group)

    # 모든 환자의 데이터를 결합
    balanced_df = pd.concat(balanced_data)

    # 결과를 CSV로 저장
    output_csv_path = os.path.join(output_save_path, 'balanced_patient_data.csv')
    balanced_df.to_csv(output_csv_path, index=False)
    print(f"Balanced data saved to {output_csv_path}")
    
    
def split_train_test_val_patient_csv(input_csv_path, output_csv_path, random_state=990912):
    """
    환자 단위로 데이터 split
    """
    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)

    # 환자 목록 가져오기
    patients = df['patient_number'].unique()

    # 환자 목록을 train, val, test로 나누기
    train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=random_state)
    val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=random_state)

    # 데이터프레임을 환자 기준으로 나누기
    train_df = df[df['patient_number'].isin(train_patients)]
    val_df = df[df['patient_number'].isin(val_patients)]
    test_df = df[df['patient_number'].isin(test_patients)]

    # 결과를 CSV로 저장
    train_df.to_csv(os.path.join(output_csv_path, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(output_csv_path, 'val_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(output_csv_path, 'test_dataset.csv'), index=False)

    print("Train, Validation, and Test datasets have been saved.")