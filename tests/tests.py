from lib import Trainables, Trainers, Observers, DataManager, BatchLosses, Samplers
from lib import DataManager as dm

import time
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

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
    
    l = Trainables.BatchLinear(n_linears, n_in, n_out, init_config={'a' : 5, 'clone' : True})
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
    
    l = Trainables.BatchLinearNoGroup(n_linears, n_in, n_out, init_method='normal', init_config={'mean' : 0.0, 'std' : 0.001})
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
    
    l1 = Trainables.BatchLinearMasked(n_linears, n_ins, n_outs, init_method='normal')
    y1 = l1(x) # (batch_size, n_linears, n_in) -> (batch_size, n_linears, n_out)
    #print(y1)
    
    l2 = Trainables.BatchLinearMasked(n_linears, n_outs, n_finals, init_method='normal')
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
    
    model = nn.Sequential(Trainables.BatchLinearNoGroup(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -INIT_RANGE, 
                                                                     'b' : INIT_RANGE}),
                          Trainables.BatchLinearNoGroup(N_NETWORKS, N_HID, N_OUT)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = BatchLosses.CrossEntropyLoss()
    
    handlers = [Observers.EnergyL1NetworkHandler(), Observers.EnergyL2NetworkHandler()]
    trackers = [Observers.Data(), 
                Observers.LossTracker(), 
                Observers.AccuracyTracker(),
                #Observers.EnergyL1NetworkTracker(),
                #Observers.EnergyL2NetworkTracker()]
                Observers.ParameterIterator(handlers=handlers)]
    
    s=time.time()
    trainer = Trainers.Trainer(model, 
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

    s=Samplers.VaryBatchAndDatasetSizeSampler(
        train, N_NETWORKS, dataset_sizes=[1, 2, 4, 8], 
        batch_sizes=1, 
        method='stretch',
        order='random'
    )
    general_collate = partial(Samplers.ensemble_collate_fn, num_networks=N_NETWORKS)
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
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 5
    BATCH_SIZE = 16
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
    
    s=Samplers.VaryBatchAndDatasetSizeSampler(
        train, N_NETWORKS, dataset_sizes=[60_000, 60_000, 60_000, 60_000, 60_000], 
        batch_sizes=[1, 1, 1, 1, 1], 
        method='loop',
        order='identical'
    )
    general_collate = Samplers.collate_fn(N_NETWORKS)
    
    samples_used = s.get_samples_per_network()

    train_dataloader = DataLoader(train,
                                  #batch_size=1)
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
    
    model = nn.Sequential(Trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(784),
                                                                     'b' : 1/np.sqrt(784),
                                                                     'clone' : True}),
                          Trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(100),
                                                                     'b' : 1/np.sqrt(100),
                                                                     'clone' : True})).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    criterion1 = BatchLosses.MSELoss(per_sample=True, reduction='mean')
    
    previous_param_provider = Observers.PreviousParameterProvider()
    initial_param_provider = Observers.InitialParameterProvider()
    prev_prev_param_provider = Observers.PreviousPreviousParameterProvider(previous_param_provider)
    handlers = [Observers.EnergyL2NeuronwiseHandler(previous_param_provider, ['incoming', 'outgoing']), 
                Observers.MinimumEnergyL2NeuronwiseHandler(initial_param_provider, ['incoming', 'outgoing']),
                Observers.EnergyL2LayerwiseHandler(previous_param_provider),
                Observers.MinimumEnergyL2LayerwiseHandler(initial_param_provider)]
    
    trackers = [Observers.Data(), 
                Observers.RunningLossTracker(),
                Observers.RunningAccuracyTracker(),
                Observers.TestingLossTracker(['test'], ['MSELoss']),
                Observers.Timer(),
                Observers.ForwardItemCounter(),
                Observers.ForwardPassCounter(),
                Observers.TestingAccuracyTracker(['test']),
                prev_prev_param_provider,
                previous_param_provider,
                initial_param_provider,
                Observers.L1Regularizer([0.0, 1e-6, 1e-5, 1e-4, 1e-3], prev_prev_param_provider),
                Observers.BackwardPassCounter(),
                Observers.BackwardItemCounter(),
                Observers.PerSampleBackwardCounter(60_000),
                Observers.PerNetworkLearningRate([0.01, 0.01, 0.01, 0.01, 0.01]),
                Observers.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : BatchLosses.MSELoss(per_sample=False, 
                                                                                                 reduction='sum')}, 
                                   device=DEVICE),
                Observers.EnergyL1NetworkTracker(previous_param_provider), # need handlers too
                #Observers.MinimumEnergyL1NeuronwiseTracker(initial_param_provider, ['incoming', 'outgoing']),
                #Observers.EnergyL2NetworkTracker(previous_param_provider),
                Observers.MinimumEnergyL1NetworkTracker(initial_param_provider)]
                #Observers.MinimumEnergyL2NetworkTracker(initial_param_provider)]
                #Observers.ParameterIterator(handlers=handlers)]
    
    s=time.time()
    
    #print(list(model.parameters()))
    
    trainer = Trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(0.2, 0.05, sample_increment=1)
    file_path = os.path.join(outputs_dir, 'experiment1.json')
    trainer.save_data_as_json(file_path)
    # acc = np.array(trainer.state['data']['test_accuracies']).T
    # en = np.array(trainer.state['data']['energies_l1']).T
    # print(acc.shape)
    # print(en.shape)
    print(time.time()-s)
    # # pal = plt.get_cmap('Reds_r', N_NETWORKS+2)
    # # for line in range(N_NETWORKS):
    # #     plt.plot(acc[line], en[line], color=pal(line))
    # # print(time.time()-s)
    # plt.show()
    # for line in range(N_NETWORKS):
    #     plt.plot(acc[line], color=pal(line))
    # plt.show()
    ## BP ##
    # model = nn.Sequential(Trainables.BatchLinearNoGroup(N_NETWORKS, N_IN, N_HID, 
    #                                                     activation=nn.GELU(),
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -0.005,
    #                                                                  'b' : 0.005,
    #                                                                  'clone' : False}),
    #                       Trainables.BatchLinearNoGroup(N_NETWORKS, N_HID, N_OUT,
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -0.005,
    #                                                                  'b' : 0.005,
    #                                                                  'clone' : False})).to(DEVICE)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # criterion = BatchLosses.MSELoss()
    
    # provider = Observers.PreviousParameterProvider()
    # handlers = [Observers.EnergyL1NetworkHandler(provider), Observers.EnergyL2NetworkHandler(provider)]
    # trackers = [Observers.Data(), 
    #             Observers.RunningLossTracker(),
    #             Observers.RunningAccuracyTracker(),
    #             Observers.TestingLossTracker(['MSELoss']),
    #             Observers.TestingAccuracyTracker(),
    #             provider,
    #             Observers.TestLoop(test_dataloader, criterions={'MSELoss' : BatchLosses.MSELoss()}, device=DEVICE),
    #             #Observers.EnergyL1NetworkTracker(),
    #             #Observers.EnergyL2NetworkTracker()]
    #             Observers.ParameterIterator(handlers=handlers)]
    
    # s=time.time()
    
    # #print(list(model.parameters()))
    
    # trainer = Trainers.Trainer(model, 
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
