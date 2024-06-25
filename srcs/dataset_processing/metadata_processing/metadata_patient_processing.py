import os
import random

import pandas as pd
from natsort import natsorted
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
    