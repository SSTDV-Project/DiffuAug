import os
import random

from torch.utils.data import Dataset
from PIL import Image


class DukeDataset(Dataset):
    def __init__(self, data_dir, transform, target_label=None, each_total=None):
        """
        Args:_
            target_label (str, optional): 'pos' or 'neg'. 만약 all이라면 모든 레이블의 데이터를 결합.
            each_total (int, optional): 각 클래스별로 포함될 이미지 개수. None이면 모든 이미지를 포함.
        """
        
        self.data_dir = data_dir
        self.transform = transform if transform is not None else None
        self.imgs = list()
        self.target_label = target_label
        
        if target_label == "all":
            self.each_total = each_total
            
        self.combine_data()
    
    def combine_data(self):
        print("Combine Duke dataset labels")

        if self.target_label == "all":
            imgs = list()
            # neg: 0, pos: 1
            for target, target_label in enumerate(['neg', 'pos']):
                case_dir = os.path.join(self.data_dir, target_label)
                print("Total target_label: ", target_label, len(os.listdir(case_dir)))

                img_files = [f for f in os.listdir(case_dir) if f.endswith('.png')]
                if self.each_total is not None:
                    img_files = random.sample(img_files, self.each_total)
                    print("Each total: ", len(img_files))
                
                for fname in img_files:
                    file_path = os.path.join(case_dir, fname)
                    imgs.append((file_path, target))
                
                # for fname in os.listdir(case_dir):
                #     if '.png' in fname:
                #         file_path = os.path.join(case_dir, fname)
                #         imgs.append((file_path, target))
        else:
            imgs = list()
            target_label_num = 1 if self.target_label == 'pos' else 0
            
            case_dir = os.path.join(self.data_dir, self.target_label)
            print("target_label: ", self.target_label, len(os.listdir(case_dir)))
            
            for fname in os.listdir(case_dir):
                if '.png' in fname:
                    file_path = os.path.join(case_dir, fname)
                    imgs.append((file_path, target_label_num))
                    
        self.imgs = imgs
    
    def __getitem__(self, index):
        FILE_PATH_POS = 0
        LABEL_POS = 1
        
        img_path = os.path.join(self.data_dir, self.imgs[index][FILE_PATH_POS])
        img_arr = Image.open(img_path)
        
        if self.transform is not None:
            data = self.transform(img_arr)
        else:
            raise Exception("Transform does not exist.")
        
        return (data, self.imgs[index][LABEL_POS])


    def __len__(self):
        return len(self.imgs)
    

