import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

def set_random_seed(seed=990912):
    random.seed(seed)

def make_metadata(data_dir, output_csv_path, output_csv_name, option="total", class_num_data=None):
    """
    Duke data 폴더의 경로를 받아 label이 분리된 metadata를 생성하는 함수.
    option이 "total"이면 전체 데이터에 대한 metadata를 생성하고,
    option이 "balanced"이면 각 label별로 지정된 개수의 데이터만 사용하여 metadata를 생성한다.
    """
    set_random_seed()
    
    print("Making metadata...")

    # 총 데이터를 저장할 데이터 프레임 생성
    columns = ['data_path', 'label']
    labeled_df = pd.DataFrame(columns=columns)
    
    if option == "total":
        output_csv_path = os.path.join(output_csv_path, output_csv_name)

        for idx, label in enumerate(['neg', 'pos']):
            data_path = os.path.join(data_dir, label)
            for f in os.listdir(data_path):
                df = pd.DataFrame(columns=columns)
                file_name = os.path.join(data_path, f)
                df['data_path'] = [file_name]
                df['label'] = [label]
                labeled_df = pd.concat([labeled_df, df], ignore_index=True)
        
        labeled_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    elif option == "balanced":
        output_csv_path = os.path.join(output_csv_path, output_csv_name)
        
        for idx, label in enumerate(['neg', 'pos']):
            data_path = os.path.join(data_dir, label)
            data_list = os.listdir(data_path)
            random.shuffle(data_list)  # 데이터 리스트를 랜덤하게 섞음
            data_list = data_list[:class_num_data]  # 각 레이블 별로 2600개 선택
            print("Data list: ", data_list)
            for f in data_list:
                df = pd.DataFrame(columns=columns)
                file_name = os.path.join(data_path, f)
                df['data_path'] = [file_name]
                df['label'] = [label]
                labeled_df = pd.concat([labeled_df, df], ignore_index=True)
        
        labeled_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')    
    
    print("Label neg length:", len(labeled_df[labeled_df['label'] == 'neg']))
    print("Label pos length:", len(labeled_df[labeled_df['label'] == 'pos']))
    print("Total length:", len(labeled_df))
    print("Metadata is saved at", output_csv_path)


def split_train_test_val_csv(meta_csv_path, csv_output_root_path):
    """
    Data의 metadata를 train, validation, test로 분할하는 함수
    """
    set_random_seed()
    
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
    

def origin_plus_augdata(origin_train_dataset_path, aug_file_parent_path, output_csv_path):
    # 1. 기존의 train_dataset.csv 파일을 불러오기
    train_df = pd.read_csv(origin_train_dataset_path)
    
    # 2. neg 및 pos 디렉토리에서 각각 랜덤으로 50개의 이미지 경로 추출
    neg_dir = os.path.join(aug_file_parent_path, 'neg')
    pos_dir = os.path.join(aug_file_parent_path, 'pos')

    neg_images = random.sample([os.path.join(neg_dir, img) for img in os.listdir(neg_dir)], 50)
    pos_images = random.sample([os.path.join(pos_dir, img) for img in os.listdir(pos_dir)], 50)
    
    # 3. 추출한 이미지 경로와 레이블을 기존 데이터에 추가
    new_data = []
    for img in neg_images:
        new_data.append({'data_path': img, 'label': 'neg'})

    for img in pos_images:
        new_data.append({'data_path': img, 'label': 'pos'})

    # 새로운 데이터를 DataFrame으로 변환
    new_df = pd.DataFrame(new_data)
    
    # 기존 데이터와 결합
    augmented_df = pd.concat([train_df, new_df], ignore_index=True)

    # 5. 새로운 aug_train_dataset.csv로 저장
    augmented_csv_path = os.path.join(output_csv_path, 'aug_train_dataset.csv')
    augmented_df.to_csv(augmented_csv_path, index=False)

    print(f"새로운 CSV 파일이 저장되었습니다: {augmented_csv_path}")
    

def test_leak_data(train_data_path, val_data_path, test_data_path):
    """
    Test data에 대한 leak 여부를 확인하는 함수
    """
    set_random_seed()
    
    print("Test data leak check...")
    
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)
    
    train_data_path = set(train_data['data_path'])
    val_data_path = set(val_data['data_path'])
    test_data_path = set(test_data['data_path'])

    print("Train data path length:", len(train_data_path))
    print("Validation data path length:", len(val_data_path))
    print("Test data path length:", len(test_data_path))
    
    print("Train and Validation data path intersection:", len(train_data_path.intersection(val_data_path)))
    print("Train and Test data path intersection:", len(train_data_path.intersection(test_data_path)))
    print("Validation and Test data path intersection:", len(val_data_path.intersection(test_data_path)))
    
    print("Test data leak check is done.")