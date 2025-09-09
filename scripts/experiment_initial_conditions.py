from lib import trainables, trainers, interceptors, batch_losses, batch_optimizers, samplers, interceptors

from lib import data_manager as dm

from lib.utils import pluck_masked_values

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
    
    RESOLUTION = 25
    INIT1_RANGE = np.logspace(np.log10(0.0005), np.log10(0.1), RESOLUTION)
    INIT2_RANGE = np.logspace(np.log10(0.0005), np.log10(1), RESOLUTION)
    
    LAYER1_INIT, LAYER2_INIT = np.meshgrid(INIT1_RANGE, INIT2_RANGE)
    LAYER1_INIT = LAYER1_INIT.reshape(-1)
    LAYER2_INIT = LAYER2_INIT.reshape(-1)

    N_NETWORKS = len(LAYER1_INIT)
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
    
    s=samplers.RandomSampler(train, 1, N_NETWORKS)
    general_collate = samplers.collate_fn(N_NETWORKS) # used to provide samples as expected for training (x, y, idx)
                                                      # tracking indices is non-standard default PyTorch 

    train_dataloader = DataLoader(train,
                              num_workers=0,
                              batch_sampler=s, # required
                              collate_fn=general_collate) # required

    test_dataloader = DataLoader(test,
                              batch_size=4096,
                              num_workers=0,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,   # lower bound (also works with lists)
                                                                     'b' : 0.01}),  # higher bound
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -0.1,
                                                                     'b' : 0.1})).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # works with torch optim
    criterion1 = batch_losses.CrossEntropyLoss(per_sample=True, reduction='mean') # note batch_losses
    
    previous_param_provider = interceptors.PreviousParameterProvider()
    trackers = [interceptors.Data(),  #      name of test loop 
                                      #            |
                interceptors.TestingLossTracker(['test'], ['MSELoss']), # <- name of criterion(s) used in test loop
                interceptors.Timer(), # tracks time
                interceptors.TestingAccuracyTracker(['test']),
                previous_param_provider, # <- should be listed before other modules that require it
                interceptors.TestLoop('test', # <- name of test loop used above
                                   test_dataloader, 
                                                                                  # per_sample is used for training
                                                                                  # sum used since we want average 
                                      #  name of criterion used above             # sample and this may change with
                                      #           |                               # batch size
                                   criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, reduction='sum')}, 
                                   device=DEVICE),
                # we attach the module keeping track of params (saves memory use with other trackers)
                #                                        |
                interceptors.EnergyL1NetworkTracker(previous_param_provider)]

    
    s=time.time()
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(0.1, 0.01)
    plt.figure(dpi=240)
    # function to grab indices that meet a condition in one array and use those indices to pull data from another array
    res = pluck_masked_values(np.array(trainer.state['data']['test_accuracies']['test']), # filter from
                              np.array(trainer.state['data']['energies_l1']), # fetch idx at filter
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