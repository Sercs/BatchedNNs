from lib import trainables, trainers, interceptors, batch_losses, batch_optimizers, samplers, utils
from lib import data_manager as dm

import time
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import copy

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

from functools import partial

import os 
import sys

# Get the absolute path of the directory containing the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Go up one directory to reach the project root
project_root = os.path.join(current_script_path, '..')

# Add the project root to the Python path
sys.path.insert(0, project_root)
outputs_dir = os.path.join(project_root, 'outputs')


torch.manual_seed(1337)

# This file contains mostly messing around for test-cases and ensuring 
# expected behavior

# TODO: low-rank residual

################################# MODEL TESTS #################################

# slower but idiomatic
def linear_batch_forward_shape():
    batch_size = 5
    n_linears = 3
    n_in = 10
    n_out = 2
        # we unsqueeze in BatchLinear forward along n_linear
        #                      |
        #                      V
    x = torch.ones(batch_size, 1, n_in)
    
    l = trainables.BatchLinear(n_linears, n_in, n_out, init_config={'a' : 5, 'clone' : True})
    y = l(x) # (batch_size, n_linears, n_in) -> (batch_size, n_linears, n_out)
    
    return y.shape == (batch_size, n_linears, n_out)

# faster (about 2x) but less intuitive
def linear_batch_no_group_forward_shape():
    batch_size = 5
    n_linears = 3
    n_in = 10
    n_out = 2
        # we unsqueeze in BatchLinear forward along n_linear
        #                      |
        #                      V
    x = torch.ones(batch_size, 1, n_in)
    
    l = trainables.BatchLinearNoGroup(n_linears, n_in, n_out, init_method='normal', init_config={'mean' : 0.0, 'std' : 0.001})
    y = l(x) # (batch_size, n_linears, n_in) -> (batch_size, n_linears, n_out)
    
    return y.shape == (batch_size, n_linears, n_out)

def linear_batch_masked_forward_shape():
    batch_size = 5
    n_linears = 4
    n_in = 10
    n_out = 6
    
    n_ins = [10, 10, 10, 10]
    n_outs = [3, 4, 5, 6]
    n_finals = [7, 6, 5, 4]
    
        # we unsqueeze in BatchLinear forward along n_linear
        #                      |
        #                      V
    x = torch.ones(batch_size, 1, n_in)
    
    l1 = trainables.BatchLinearMasked(n_linears, n_ins, n_outs, init_method='normal')
    y1 = l1(x) # (batch_size, n_linears, n_in) -> (batch_size, n_linears, n_out)
    #print(y1)
    
    l2 = trainables.BatchLinearMasked(n_linears, n_outs, n_finals, init_method='normal')
    y2 = l2(y1)
    
    print(y2.mean((0, -1)).shape)
    print(y2.shape)
    
def dataset_test():#
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
    print(len(train))
    
def train_test():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 1
    BATCH_SIZE = 16
    N_IN = 784
    N_HID = 20
    N_OUT = 10
    INIT_RANGE = np.linspace(0.005, 0.5, N_NETWORKS)

    
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
    
    train_dataloader = DataLoader(train, 
                              batch_size=BATCH_SIZE)

    test_dataloader = DataLoader(test,
                              batch_size=1_000,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinearNoGroup(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -INIT_RANGE, 
                                                                     'b' : INIT_RANGE}),
                          trainables.BatchLinearNoGroup(N_NETWORKS, N_HID, N_OUT)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = batch_losses.CrossEntropyLoss()
    
    handlers = [interceptors.EnergyL1NetworkHandler(), interceptors.EnergyL2NetworkHandler()]
    trackers = [interceptors.Data(), 
                interceptors.LossTracker(), 
                interceptors.AccuracyTracker(),
                #interceptors.EnergyL1NetworkTracker(),
                #interceptors.EnergyL2NetworkTracker()]
                interceptors.ParameterIterator(handlers=handlers)]
    
    s=time.time()
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(3.0, 0.05)
    
    acc = np.array(trainer.state['data']['test_accuracies']).T
    en = np.array(trainer.state['data']['energies_l2']).T
    print(acc.shape)
    print(en.shape)
    for line in range(N_NETWORKS):
        plt.plot(acc[line], en[line])
    print(time.time()-s)
    
#if __name__ == "__main__":
def sampler_samples_test():
    N_NETWORKS = 4

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
    
    train = dm.DatasetWithIdx(train_dataset, task='classify')

    s=samplers.VaryBatchAndDatasetSizeSampler(
        train, N_NETWORKS, dataset_sizes=[1, 2, 4, 8], 
        batch_sizes=1, 
        method='stretch',
        order='random'
    )
    general_collate = partial(samplers.ensemble_collate_fn, num_networks=N_NETWORKS)
    samples_used = s.get_samples_per_network()
    train_dataloader = DataLoader(train,
                                  #batch_size=1)
                              num_workers=0,
                              batch_sampler=s,
                              collate_fn=general_collate)
    for i in train_dataloader:
        print(i[-1])
    print(samples_used)

#def sampler_test():
if __name__ == '__main__':
    #data = []
    #for DEGREES in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 1
    BATCH_SIZE = 1
    N_IN = 784
    N_HID = 100
    N_OUT = 10
    DEGREES = np.logspace(-1, 1.5, N_NETWORKS-1)
    DEGREES = np.append(2, DEGREES)
    LR = 0.01 #np.logspace(-5, -3, N_NETWORKS)


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
    
    s=samplers.IdenticalSampler(train, BATCH_SIZE, N_NETWORKS)
    general_collate = samplers.collate_fn(N_NETWORKS)
    
    #samples_used = s.get_samples_per_network()

    train_dataloader = DataLoader(train,
                                  #batch_size=1)
                              pin_memory=True,
                              num_workers=0,
                              batch_sampler=s,
                              collate_fn=general_collate)
    
    eval_dataloader = DataLoader(train,
                                 batch_size=16,
                                 num_workers=4,
                                 shuffle=False)
    print(len(train_dataloader))
    test_dataloader = DataLoader(test,
                              batch_size=16,
                              num_workers=4,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(784),
                                                                     'b' : 1/np.sqrt(784)}
                                                        ),
                          # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
                          #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                          #                               mu_lr=np.linspace(0.01, 0.1, 5)),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(100),
                                                                     'b' : 1/np.sqrt(100)}
                                                        ),
                          ).to(DEVICE)
    
    optimizer = batch_optimizers.SGD(model.parameters(), lr=0.01)
    # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    #                                                         reduction='mean'),
    #                                    per_sample=True,
    #                                    reduction='mean') # note batch_losses
    
    criterion1 = batch_losses.MSELoss(per_sample=True, 
                                      reduction='mean')

    previous_param_provider = interceptors.PreviousParameterProvider()
    initial_param_provider = interceptors.InitialParameterProvider()
    prev_prev_param_provider = interceptors.PreviousPreviousParameterProvider(previous_param_provider)
    
    
    handlers = [interceptors.EnergyL0NetworkHandler()]
    trackers = [interceptors.Timer(),
                interceptors.TestingLossTracker({'test': ['MSELoss']}),
                interceptors.TestingAccuracyTracker(['test']),
                previous_param_provider,
                initial_param_provider,
                interceptors.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
                                                                                reduction='sum')}, 
                                   device=DEVICE),
                interceptors.BackwardPassCounter(),
                #interceptors.EnergyL0NetworkTracker(),
                interceptors.EnergyL1NetworkTracker(previous_param_provider),
                interceptors.EnergyL1LayerwiseTracker(previous_param_provider),
                #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                interceptors.ParameterIterator(handlers)]
    
    s=time.time()
    
    #print(list(model.parameters()))
    
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(0.05, 0.01, sample_increment=1)
    
    d=utils.convert_data(trainer.state['data'])
        #data.append(copy.deepcopy(d))

    ## BP ##
    # model = nn.Sequential(trainables.BatchLinearNoGroup(N_NETWORKS, N_IN, N_HID, 
    #                                                     activation=nn.GELU(),
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -0.005,
    #                                                                  'b' : 0.005,
    #                                                                  'clone' : False}),
    #                       trainables.BatchLinearNoGroup(N_NETWORKS, N_HID, N_OUT,
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -0.005,
    #                                                                  'b' : 0.005,
    #                                                                  'clone' : False})).to(DEVICE)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # criterion = batch_losses.MSELoss()
    
    # provider = interceptors.PreviousParameterProvider()
    # handlers = [interceptors.EnergyL1NetworkHandler(provider), interceptors.EnergyL2NetworkHandler(provider)]
    # trackers = [interceptors.Data(), 
    #             interceptors.RunningLossTracker(),
    #             interceptors.RunningAccuracyTracker(),
    #             interceptors.TestingLossTracker(['MSELoss']),
    #             interceptors.TestingAccuracyTracker(),
    #             provider,
    #             interceptors.TestLoop(test_dataloader, criterions={'MSELoss' : batch_losses.MSELoss()}, device=DEVICE),
    #             #interceptors.EnergyL1NetworkTracker(),
    #             #interceptors.EnergyL2NetworkTracker()]
    #             interceptors.ParameterIterator(handlers=handlers)]
    
    # s=time.time()
    
    # #print(list(model.parameters()))
    
    # trainer = trainers.Trainer(model, 
    #                            N_NETWORKS, 
    #                            optimizer, 
    #                            criterion, 
    #                            train_dataloader, 
    #                            test_dataloader, 
    #                            trackers=trackers, 
    #                            device=DEVICE)
    # trainer.train_loop(0.5, 0.2)
    # acc2 = np.array(trainer.state['data']['test_accuracies']).T
    # en2 = np.array(trainer.state['data']['energies_l1']).T
    # print(acc2.shape)
    # print(en2.shape)
    # pal2 = plt.get_cmap('Blues_r', N_NETWORKS+2)
    # for line in range(N_NETWORKS):
    #     #plt.plot(acc[line], en[line], color=pal(line))
    #     plt.plot(acc2[line], en2[line], color=pal2(line))
    # plt.yscale('log')
    # print(time.time()-s)
    # plt.show()
    # for line in range(N_NETWORKS):
    #     plt.plot(acc2[line], color=pal2(line))
    # plt.show()
    
def recur_test():
#if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 50
    BATCH_SIZE = 1
    N_IN = 784
    N_HID = 100
    N_OUT = 10
    INIT_RANGE = np.linspace(0.005, 0.5, N_NETWORKS)

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
    general_collate = samplers.collate_fn(N_NETWORKS)
    
    #samples_used = s.get_samples_per_network()

    train_dataloader = DataLoader(train,
                                  #batch_size=1)
                              pin_memory=True,
                              num_workers=4,
                              batch_sampler=s,
                              collate_fn=general_collate)
    
    eval_dataloader = DataLoader(train,
                                 batch_size=16,
                                 num_workers=4,
                                 shuffle=False)
    print(len(train_dataloader))
    test_dataloader = DataLoader(test,
                              batch_size=16,
                              num_workers=4,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(784),
                                                                     'b' : 1/np.sqrt(784)}),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(100),
                                                                     'b' : 1/np.sqrt(100)}),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(100),
                                                                     'b' : 1/np.sqrt(100)})
                          ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.0)
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean')
    
    previous_param_provider = interceptors.PreviousParameterProvider()
    initial_param_provider = interceptors.InitialParameterProvider()

    trackers = [interceptors.TestingLossTracker({'test': ['MSELoss']}),
                interceptors.Timer(),
                interceptors.TestingAccuracyTracker(['test']),
                previous_param_provider,
                initial_param_provider,
                interceptors.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
                                                                                                 reduction='sum')}, 
                                   device=DEVICE),
                interceptors.EnergyL1NetworkTracker(previous_param_provider), # need handlers too
                interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider)]
    
    s=time.time()
    
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(10, 0.05, sample_increment=1)
    d1 = trainer.state['data'].copy()
    print(np.array(trainer.state['data']['test_accuracies']['test'])) # TODO: might need "fire on init"
    model[1].n_recurs = 1
    new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.0)
    trainer.state['optimizer'] = new_optimizer
    trainer.train_loop(10, 0.05, sample_increment=1)
    print(np.array(trainer.state['data']['test_accuracies']['test']))
    d2 = trainer.state['data']
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(784),
                                                                     'b' : 1/np.sqrt(784)}),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(100),
                                                                     'b' : 1/np.sqrt(100)}),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(100),
                                                                     'b' : 1/np.sqrt(100)})
                          ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.0)
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean')
    
    previous_param_provider = interceptors.PreviousParameterProvider()
    initial_param_provider = interceptors.InitialParameterProvider()

    trackers = [interceptors.TestingLossTracker({'test': ['MSELoss']}),
                interceptors.Timer(),
                interceptors.TestingAccuracyTracker(['test']),
                previous_param_provider,
                initial_param_provider,
                interceptors.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
                                                                                                 reduction='sum')}, 
                                   device=DEVICE),
                interceptors.EnergyL1NetworkTracker(previous_param_provider), # need handlers too
                interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider)]
    
    s=time.time()
    
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(10, 0.05, sample_increment=1)
    d3 = trainer.state['data'].copy()
    print(np.array(trainer.state['data']['test_accuracies']['test'])) # TODO: might need "fire on init"
    new_model = nn.Sequential(model[0], 
                              model[1], 
                              trainables.BatchLinear(N_NETWORKS, N_HID, N_HID, 
                                                            activation=nn.GELU(),
                                                            init_method='uniform',
                                                            init_config={'a' : -1/np.sqrt(100),
                                                                         'b' : 1/np.sqrt(100)}),
                              model[2]
                              ).to(DEVICE)
    previous_param_provider = interceptors.PreviousParameterProvider()
    initial_param_provider = interceptors.InitialParameterProvider()

    trackers = [interceptors.TestingLossTracker({'test': ['MSELoss']}),
                interceptors.Timer(),
                interceptors.TestingAccuracyTracker(['test']),
                previous_param_provider,
                initial_param_provider,
                interceptors.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
                                                                                                 reduction='sum')}, 
                                   device=DEVICE),
                interceptors.EnergyL1NetworkTracker(previous_param_provider), # need handlers too
                interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider)]
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.00025, weight_decay=0.0)
    trainer = trainers.Trainer(new_model, 
                               N_NETWORKS, 
                               new_optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(10, 0.05, sample_increment=1)
    print(np.array(trainer.state['data']['test_accuracies']['test']))
    d4 = trainer.state['data']

    

# TODO: Make trainer even more general, more observers (i.e. counts), lazy losses, data, init
    
################################## PRINTOUTS ##################################

#sampler_test()

#dataset_test()

# linear_batch_masked_forward_shape()

# N = 10_000
# start1 = time.time()
# print(linear_batch_forward_shape())
# for i in range(N):
#     linear_batch_forward_shape()
# end1 = time.time()
# grouped_time = end1 - start1

# start2 = time.time()
# print(linear_batch_no_group_forward_shape())
# for i in range(N):
#     linear_batch_no_group_forward_shape()
# end2 = time.time()
# ungrouped_time = end2 - start2
# print(f"Grouped Params Time: {grouped_time}\n" +
#       f"Ungrouped Params. Time: {ungrouped_time}\n" +
#       f"Speed up: {grouped_time/ungrouped_time:.2f}x")


"""
    I want something that could be used to create subdatasets that can be created at the end of an epoch.
    This requires computing the samples StatefulLazy uses to update on. 
    We could have an observer that tracks this memory by looking at backwards updates. With a batch size
    we need to track individual samples! 
    
    I also want to track "samples required" or "samples forwarded on", along with forward pass counts 
    (noting -1 idx for small datasets) 
"""
