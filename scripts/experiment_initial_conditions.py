from lib import Trainables, Trainers, Observers, BatchLosses, Samplers
from lib import DataManager as dm

from lib.Utils import pluck_masked_values

import time

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    RESOLUTION = 50
    INIT1_RANGE = np.logspace(np.log10(0.0005), np.log10(0.1), RESOLUTION)
    INIT2_RANGE = np.logspace(np.log10(0.0005), np.log10(1), RESOLUTION)
    
    LAYER1_INIT, LAYER2_INIT = np.meshgrid(INIT1_RANGE, INIT2_RANGE)
    LAYER1_INIT = LAYER1_INIT.reshape(-1)
    LAYER2_INIT = LAYER2_INIT.reshape(-1)

    N_NETWORKS = 3# len(LAYER1_INIT)
    N_IN = 784
    N_HID = 100
    N_OUT = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='datasets',
                             #split='digits', # used as a quick swap for EMNIST
                                train=True,
                                download=True,
                                transform=transform, # automatically coverts 0 - 255 --> 0 - 1
                                target_transform=dm.temp_onehot)
    test_dataset = datasets.MNIST(root='datasets',
                             #split='digits', # used as a quick swap for EMNIST
                                train=False,
                                download=True,
                                transform=transform, # automatically coverts 0 - 255 --> 0 - 1
                                target_transform=dm.temp_onehot)
    
    train = dm.DatasetWithIdx(train_dataset, task='classify')
    test = dm.DatasetWithIdx(test_dataset, task='classify')
    
    s=Samplers.RandomSampler(train, 1, N_NETWORKS)
    general_collate = Samplers.collate_fn(N_NETWORKS) # used to provide samples as expected for training (x, y, idx)
                                                      # tracking indices is non-standard default PyTorch 

    train_dataloader = DataLoader(train,
                              num_workers=4,
                              batch_sampler=s,
                              pin_memory=True,
                              collate_fn=general_collate)

    test_dataloader = DataLoader(test,
                              batch_size=16,
                              num_workers=4,
                              shuffle=False)
    
    model = nn.Sequential(Trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1, # works with lists
                                                                     'b' : 1}),
                          Trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1,
                                                                     'b' : 1})).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion1 = BatchLosses.CrossEntropyLoss(per_sample=True, reduction='mean')
    
    previous_param_provider = Observers.PreviousParameterProvider()
    trackers = [Observers.Data(),  #      name of test loop 
                                   #            |
                Observers.TestingLossTracker(['test'], ['MSELoss']), # <- name of criterion(s) used in test loop
                Observers.Timer(), # tracks time
                Observers.TestingAccuracyTracker(['test']),
                previous_param_provider,
                Observers.TestLoop('test', # <- name of test loop used above
                                   test_dataloader, 
                                            # name of criterion used above        # per_sample is used for training
                                            #      |                              # sum used since we want average 
                                            #      |                              # sample and this may change with
                                            #      |                              # batch size
                                   criterions={'MSELoss' : BatchLosses.MSELoss(per_sample=False, reduction='sum')}, 
                                   device=DEVICE),
                # we attach the module keeping track of params (saves memory use with other trackers)
                #                                        |
                Observers.EnergyL1NetworkTracker(previous_param_provider)]

    
    s=time.time()
    trainer = Trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(3.0, 0.01)
    plt.figure(dpi=240)
    res = pluck_masked_values(np.array(trainer.state['data']['test_accuracies']['test']).T, # filter from
                              np.array(trainer.state['data']['energies_l1']).T, # fetch idx at filter
                              lambda x : x > 9000) # find first value over 9000 (i.e. 90% test accuracy)
    plt.contourf(LAYER1_INIT.reshape(RESOLUTION, RESOLUTION), 
                 LAYER2_INIT.reshape(RESOLUTION, RESOLUTION), 
                 res.reshape(RESOLUTION, RESOLUTION), 
                 levels=10, 
                 cmap='viridis')
    plt.xlabel('Layer 1 Initialization Range')
    plt.ylabel('Layer 2 Initialization Range')
    plt.xscale('log')
    plt.yscale('log')
    
    
    #file_path = './experiment_initial_conditions.json'
    #trainer.save_data_as_json(file_path)
    print(time.time()-s)