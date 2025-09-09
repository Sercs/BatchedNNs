# samplers
# dataset
# save/load
# expects data in python types

import torch
from torch.utils.data import Dataset

from torchvision.transforms import ToTensor
from torchvision import datasets

from torch.utils.data import Sampler
from torch.utils.data import Subset

import json
import numpy as np

def save_data(file_name, data):
    with open(file_name + ".json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

# loads data into default python types
def load_data(file_name):
    with open(file_name + ".json", "r") as f:
        data = json.load(f)
    return data

# convert from python list to numpy array because lists don't support math :(
# i.e. typically used like convert_data(load_data('path/data'))
def convert_data(data_dict):
    for key in data_dict:
        if type(data_dict[key]) == list:
            data_dict[key] = np.array(data_dict[key])
    return data_dict

class DatasetWithIdx(Dataset):
    def __init__(self, dataset, task='classify', padding_value=-1):
        super(DatasetWithIdx, self).__init__()
        self.dataset = dataset   
        self.task = task
        self.padding_value = padding_value
        
        # Placeholders for masked samples when using varied dataset_sizes
        # or varied batch_sizes.
        sample_x, sample_y = self.dataset[0]
        self.x_placeholder = torch.zeros_like(sample_x.flatten())
        self.y_placeholder = torch.zeros_like(sample_y)
        
    def __getitem__(self, index):
        if 'clas' in self.task or 'reg' in self.task:
            if index == self.padding_value:
                return self.x_placeholder, self.y_placeholder, self.padding_value
            x, y = self.dataset[index]
            return x.flatten(), y, index
        
        elif 'auto' in self.task or 'ae' in self.task:
            if index == self.padding_value:
                return self.x_placeholder, self.x_placeholder, self.padding_value
            x, _ = self.dataset[index]
            return x.flatten(), x.flatten(), index
        
        else:
            raise Exception('Task not implemented!')
     
    def __len__(self):
        return(len(self.dataset))
        

def temp_onehot(y, n_classes=10):
    return torch.zeros(n_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

class MNIST(Dataset):
    def __init__(self, train=True, use_emnist=False, transform=None, num_classes=10, device='cpu'):
        if transform is None:
            transform = ToTensor()
        if use_emnist:
            self.dataset = datasets.EMNIST(root='datasets',
                                        split='digits', # used as a quick swap for EMNIST
                                        train=train,
                                        download=True,
                                        transform=transform, # automatically coverts 0 - 255 --> 0 - 1
                                        target_transform=temp_onehot)
    
        else:
            self.dataset = datasets.MNIST(root='datasets',
                                     #split='digits', # used as a quick swap for EMNIST
                                        train=train,
                                        download=True,
                                        transform=transform, # automatically coverts 0 - 255 --> 0 - 1
                                        target_transform=temp_onehot)
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return(len(self.dataset))