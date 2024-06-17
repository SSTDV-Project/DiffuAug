import os

from torch.utils.data import Dataset
from PIL import Image


class DukeDataset(Dataset):
    def __init__(self, data_dir, transform, target_label=None):
        """
        Args:_
            target_label (str, optional): 'pos' or 'neg'. 만약 None이라면 모든 레이블의 데이터를 결합.
        """
        
        self.data_dir = data_dir
        self.transform = transform if transform is not None else None
        self.imgs = list()
        
        if target_label is None:
            self.combine_data()
        else:
            self.target_label = target_label
            self.combine_data(self.target_label)
    
    def combine_data(self, target_label=None):
        print("Combine Duke dataset labels")

        if target_label is None:
            imgs = list()
            # neg: 0, pos: 1
            for target, target_label in enumerate(['neg', 'pos']):
                case_dir = os.path.join(self.data_dir, target_label)
                print("target_label: ", target_label, len(os.listdir(case_dir)))
                
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        file_path = os.path.join(case_dir, fname)
                        imgs.append((file_path, target))
        else:
            imgs = list()
            target_label_num = 1 if target_label == 'pos' else 0
            
            case_dir = os.path.join(self.data_dir, target_label)
            print("target_label: ", target_label, len(os.listdir(case_dir)))
            
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
    

