import os

import pandas as pd
from sklearn.model_selection import train_test_split

def make_metadata(data_dir, output_csv_path):
    """
    Duke data 폴더의 경로를 받아 label이 분리된 metadata를 생성하는 함수
    """
    
    print("Making metadata...")
    
    output_csv_path = os.path.join(output_csv_path, "duke_metadata.csv")
    colums = ['data_path', 'label']
    labeld_df = pd.DataFrame(columns=colums)
    
    for idx, label in enumerate(['neg', 'pos']):
        data_path = os.path.join(data_dir, label)
        for f in os.listdir(data_path):
            df = pd.DataFrame(columns=colums)
            file_name = os.path.join(data_path, f)
            df['data_path'] = [file_name]
            df['label'] = [label]
            labeld_df = pd.concat([labeld_df, df], ignore_index=True)
    
    labeld_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print("Label neg length:", len(labeld_df[labeld_df['label'] == 'neg']))
    print("Label pos length:", len(labeld_df[labeld_df['label'] == 'pos']))
    print("Total length:", len(labeld_df))
    print("Metadata is saved at", output_csv_path)


def split_train_test_val_csv(meta_csv_path, csv_output_root_path):
    """
    Data의 metadata를 train, validation, test로 분할하는 함수
    """
    print("Metadata splitting...")
    
    # CSV 파일 읽기
    data = pd.read_csv(meta_csv_path)

    # negative와 positive 데이터 분리
    negative_data = data[data['label'] == 'neg']
    positive_data = data[data['label'] == 'pos']

    # 데이터 셋 분할 (train: 80%, validation: 10%, test: 10%)
    neg_train, neg_temp = train_test_split(negative_data, test_size=0.2, random_state=990912)
    neg_val, neg_test = train_test_split(neg_temp, test_size=0.5, random_state=990912)

    pos_train, pos_temp = train_test_split(positive_data, test_size=0.2, random_state=990912)
    pos_val, pos_test = train_test_split(pos_temp, test_size=0.5, random_state=990912)

    # 각 세트 병합
    train_data = pd.concat([neg_train, pos_train])
    val_data = pd.concat([neg_val, pos_val])
    test_data = pd.concat([neg_test, pos_test])

    # CSV 파일로 저장
    train_csv_path = os.path.join(csv_output_root_path, 'train_dataset.csv')
    val_csv_path = os.path.join(csv_output_root_path, 'val_dataset.csv')
    test_csv_path = os.path.join(csv_output_root_path, 'test_dataset.csv')
    
    train_data.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
    val_data.to_csv(val_csv_path, index=False, encoding='utf-8-sig')
    test_data.to_csv(test_csv_path, index=False, encoding='utf-8-sig')

    print("Train dataset length:", len(train_data))
    print("Train negative length:", len(train_data[train_data['label'] == 'neg']))
    print("Train positive length:", len(train_data[train_data['label'] == 'pos']))
    print("-----------------------------------------")
    
    print("Validation dataset length:", len(val_data))
    print("Validation negative length:", len(val_data[val_data['label'] == 'neg']))
    print("Validation positive length:", len(val_data[val_data['label'] == 'pos']))
    print("-----------------------------------------")
    
    print("Test dataset length:", len(test_data))
    print("Test negative length:", len(test_data[test_data['label'] == 'neg']))
    print("Test positive length:", len(test_data[test_data['label'] == 'pos']))
    print("-----------------------------------------")
    
    print("Total length:", len(train_data) + len(val_data) + len(test_data))
    print("Total negative length:", len(train_data[train_data['label'] == 'neg']) + len(val_data[val_data['label'] == 'neg']) + len(test_data[test_data['label'] == 'neg']))
    print("Total positive length:", len(train_data[train_data['label'] == 'pos']) + len(val_data[val_data['label'] == 'pos']) + len(test_data[test_data['label'] == 'pos']))
    
    print("Metadata is splitted and saved at", csv_output_root_path)