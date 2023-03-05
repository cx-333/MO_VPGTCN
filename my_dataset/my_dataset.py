# _*_ coding:utf-8 _*_

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class My_Dataset(Dataset):
    def __init__(self, path:str, samples:list) -> None:
        super(My_Dataset, self).__init__()
        self.path, self.samples = path, samples

    def __getitem__(self, index: int) -> torch.Tensor:
        dna_path = os.path.join(self.path, 'dna', self.samples[index][0]+'.txt')
        rna_path = os.path.join(self.path, 'rna', self.samples[index][0]+'.txt')
        rppa_path = os.path.join(self.path, 'rppa', self.samples[index][0]+'.txt')
        dna_data = np.loadtxt(dna_path, delimiter=',')
        rna_data = np.loadtxt(rna_path, delimiter=',')
        rppa_data = np.loadtxt(rppa_path, delimiter=',')
        return dna_data, rna_data, rppa_data, self.samples[index][1]

    def __len__(self):
        return len(self.samples)

def get_samples(path:str) -> list:
    # samples list
    result = [ ]
    with open(path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
        for text in texts:
            temp = text.split(' ')
            result.append([temp[0], int(temp[1].replace('\n', ''))])
    return result


def load_data(batch_size:int=2) -> object:
    # yield dataloader
    train_sample_path = 'my_dataset/train.txt'
    val_sample_path = 'my_dataset/validation.txt'
    train_samples = get_samples(train_sample_path)
    validation_samples = get_samples(val_sample_path)
    train_dataset = My_Dataset('../code/data/train', train_samples)
    val_dataset = My_Dataset('../code/data/validation', validation_samples)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_data, val_data

