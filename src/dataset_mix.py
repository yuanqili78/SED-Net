import os
import sys
from src.dataset_segments import ori_simple_data
from src.dataset_segments_my import my_simple_data

from torch.utils.data import Dataset


class my_mix_dataset(Dataset):
    def __init__(self, prefix="", if_normals=False, if_train=False, aug=True, noise=False, noise_level=1):
        super().__init__()
        self.ori_data = ori_simple_data(prefix=prefix, if_normals=if_normals, if_train=if_train, aug=aug, noise=noise, noise_level=noise_level)
        self.my_data = my_simple_data(prefix=prefix, if_normals=if_normals, if_train=if_train, aug=aug, noise=noise, noise_level=noise_level)

    
    def __len__(self, ):
        return len(self.my_data) + len(self.ori_data) 

    def __getitem__(self, index):
        if index < len(self.my_data):
            return self.my_data.__getitem__(index)
        
        else:
            return self.ori_data.__getitem__(index - len(self.my_data))



