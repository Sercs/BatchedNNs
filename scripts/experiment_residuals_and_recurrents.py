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
    
    N_NETWORKS = 10
    N_IN = 784
    N_HIDs = [100, 100, 100, 100]
    N_OUT = 10
    
    N_EPOCHS = 20.0
    TEST_INTERVAL = 0.05

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
                              num_workers=1,
                              batch_sampler=s, # required
                              collate_fn=general_collate) # required

    test_dataloader = DataLoader(test,
                              batch_size=64,
                              num_workers=1,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HIDs[0], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[0], N_HIDs[1], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}), 
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[1], N_HIDs[2], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[2], N_HIDs[2], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[2], N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01})
                          ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0)
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean') # note batch_losses
    
    previous_param_provider = interceptors.PreviousParameterProvider()
                                      #      name of test loop 
                                      #            |
    trackers = [interceptors.TestingLossTracker({'test' : 'MSELoss'}), # <- name of criterion(s) used in test loop
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
    trainer.train_loop(N_EPOCHS, TEST_INTERVAL)
    d1 = trainer.state['data']
    plt.figure(dpi=240)
    
    acc1 = np.array(d1['test_accuracies']['test']).T
    en1 = np.array(d1['energies_l1']).T
    
    pal1 = plt.get_cmap('Blues_r', N_NETWORKS+1)
    for i in range(len(acc1)):
        plt.plot(acc1[i], en1[i], color=pal1(i))
    plt.show()

    ##################################################################################################################


    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HIDs[0], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[0], N_HIDs[1], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01},
                                                        add_residual=True), 
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[1], N_HIDs[2], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01},
                                                        add_residual=True),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[2], N_HIDs[3], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01},
                                                        add_residual=True), 
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[3], N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01})
                          ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0)
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean') # note batch_losses
    
    previous_param_provider = interceptors.PreviousParameterProvider()
                                      #      name of test loop 
                                      #            |
    trackers = [interceptors.TestingLossTracker({'test' : 'MSELoss'}), # <- name of criterion(s) used in test loop
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
    trainer.train_loop(N_EPOCHS, TEST_INTERVAL)
    d2 = trainer.state['data']
    plt.figure(dpi=240)
    
    acc2 = np.array(d2['test_accuracies']['test']).T
    en2 = np.array(d2['energies_l1']).T
    
    pal2 = plt.get_cmap('Reds_r', N_NETWORKS+1)
    for i in range(len(acc2)):
        plt.plot(acc2[i], en2[i], color=pal2(i))
    plt.show()
        
    ##################################################################################################################


    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HIDs[0], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[0], N_HIDs[2], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01},
                                                        n_recurs=2),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[2], N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01})
                          ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0)
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean') # note batch_losses
    
    previous_param_provider = interceptors.PreviousParameterProvider()
                                      #      name of test loop 
                                      #            |
    trackers = [interceptors.TestingLossTracker({'test' : 'MSELoss'}), # <- name of criterion(s) used in test loop
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
    trainer.train_loop(N_EPOCHS, TEST_INTERVAL)
    d3 = trainer.state['data']
    plt.figure(dpi=240)
    
    acc3 = np.array(d3['test_accuracies']['test']).T
    en3 = np.array(d3['energies_l1']).T
    
    pal3 = plt.get_cmap('Greens_r', N_NETWORKS+1)
    for i in range(len(acc3)):
        plt.plot(acc3[i], en3[i], color=pal3(i))

    plt.show()
    ##################################################################################################################


    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HIDs[0], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01}),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[0], N_HIDs[2], 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01},
                                                        n_recurs=2,
                                                        add_residual=True),
                          trainables.BatchLinear(N_NETWORKS, N_HIDs[2], N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -0.01,
                                                                     'b' : 0.01})
                          ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0) # works with torch optim
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean') # note batch_losses
    
    previous_param_provider = interceptors.PreviousParameterProvider()
                                      #      name of test loop 
                                      #            |
    trackers = [interceptors.TestingLossTracker({'test' : 'MSELoss'}), # <- name of criterion(s) used in test loop
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
    trainer.train_loop(N_EPOCHS, TEST_INTERVAL)
    d4 = trainer.state['data']
    plt.figure(dpi=240)
    
    acc4 = np.array(d4['test_accuracies']['test']).T
    en4 = np.array(d4['energies_l1']).T
    
    pal4 = plt.get_cmap('Purples_r', N_NETWORKS+1)
    for i in range(len(acc4)):
        plt.plot(acc4, en4, color=pal4(i))
    plt.show()
    plt.figure(dpi=240)
    for i in range(len(acc4)):
        plt.plot(acc1[i], en1[i], color=pal1(i))
        plt.plot(acc2[i], en2[i], color=pal2(i))
        plt.plot(acc3[i], en3[i], color=pal3(i))
        plt.plot(acc4[i], en4[i], color=pal4(i))
    plt.legend(['Multilayered', 'Residual', 'Recurrent', 'Residual+Recurrent'])
    plt.show()