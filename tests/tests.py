from lib import trainables, trainers, interceptors, batch_losses, batch_optimizers, samplers, utils, masks
from lib import data_manager as dm

from lib import batch_optimizers 

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

import gc

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

def sampler_test():
#if __name__ == '__main__':
    #data = []
    #for DEGREES in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 10
    BATCH_SIZE = 1
    N_IN = 784
    N_HID = 100
    N_OUT = 10
    DEGREES = np.logspace(-1, 1.5, N_NETWORKS-1)
    DEGREES = np.append(2, DEGREES)
    LR = np.logspace(-5, -3, N_NETWORKS)
    margins = np.linspace(0.0, 2.0, N_NETWORKS)


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
                                                        init_config={'a' : -1/np.power(784, 0.5),
                                                                     'b' :1/np.power(784, 0.5)}
                                                        ),
                          # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
                          #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                          #                               mu_lr=np.linspace(0.01, 0.1, 5)),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          ).to(DEVICE)
    
    optimizer = batch_optimizers.SGD(model.parameters(), lr=LR, momentum=LR)
    # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    #                                                         reduction='mean'),
    #                                    per_sample=True,
    #                                    reduction='mean') # note batch_losses
    
    criterion1 = batch_losses.HingeLoss(per_sample=True, 
                                      reduction='mean',
                                      func=nn.ReLU(),
                                      margin=margins)

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
                interceptors.EnergyL0NetworkTracker(),
                interceptors.EnergyL1NetworkTracker(previous_param_provider),
                interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
                #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                #interceptors.ParameterIterator(handlers)]
    
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
    trainer.train_loop(2, 0.01, sample_increment=1)
    
    d1=utils.convert_data(trainer.state['data'])
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(784, 0.5),
                                                                     'b' :1/np.power(784, 0.5)}
                                                        ),
                          # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
                          #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                          #                               mu_lr=np.linspace(0.01, 0.1, 5)),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          ).to(DEVICE)
    
    optimizer = batch_optimizers.AdamW(model.parameters(), lr=LR, beta1=LR)
    # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    #                                                         reduction='mean'),
    #                                    per_sample=True,
    #                                    reduction='mean') # note batch_losses
    
    criterion1 = batch_losses.HingeLoss(per_sample=True, 
                                        reduction='mean',
                                        func=nn.Softplus(beta=3.0),
                                        margin=margins)

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
                interceptors.EnergyL0NetworkTracker(),
                interceptors.EnergyL1NetworkTracker(previous_param_provider),
                interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
                #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                #interceptors.ParameterIterator(handlers)]
    
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
    trainer.train_loop(2, 0.01, sample_increment=1)
    
    d2=utils.convert_data(trainer.state['data'])
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(784, 0.5),
                                                                     'b' :1/np.power(784, 0.5)}
                                                        ),
                          # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
                          #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                          #                               mu_lr=np.linspace(0.01, 0.1, 5)),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          ).to(DEVICE)
    
    optimizer = batch_optimizers.AdamP(model.parameters(), lr=LR, beta2=LR)
    # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    #                                                         reduction='mean'),
    #                                    per_sample=True,
    #                                    reduction='mean') # note batch_losses
    
    criterion1 = batch_losses.StatefulLazyLoss(batch_losses.HingeLoss(per_sample=True, 
                                                                      reduction='mean',
                                                                      func=nn.Softplus(beta=3.0),
                                                                      margin=margins),
                                            max_samples=60_000,
                                            n_networks=N_NETWORKS)

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
                interceptors.EnergyL0NetworkTracker(),
                interceptors.EnergyL1NetworkTracker(previous_param_provider),
                interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
                #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                #interceptors.ParameterIterator(handlers)]
    
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
    trainer.train_loop(2, 0.01, sample_increment=1)
    
    d3=utils.convert_data(trainer.state['data'])
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(784, 0.5),
                                                                     'b' :1/np.power(784, 0.5)}
                                                        ),
                          # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
                          #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                          #                               mu_lr=np.linspace(0.01, 0.1, 5)),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          ).to(DEVICE)
    
    optimizer = batch_optimizers.LazyAdamW(model.parameters(), lr=0.0005)
    # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    #                                                         reduction='mean'),
    #                                    per_sample=True,
    #                                    reduction='mean') # note batch_losses
    
    criterion1 = batch_losses.LazyLoss(batch_losses.HingeLoss(per_sample=True, 
                                                                      reduction='mean',
                                                                      func=nn.Softplus(beta=3.0),
                                                                      margin=margins))

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
                interceptors.EnergyL0NetworkTracker(),
                interceptors.EnergyL1NetworkTracker(previous_param_provider),
                interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
                #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                #interceptors.ParameterIterator(handlers)]
    
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
    trainer.train_loop(2, 0.01, sample_increment=1)
    
    d4=utils.convert_data(trainer.state['data'])
    
    pal = [plt.get_cmap('Greys', 10), plt.get_cmap('Reds', 10), plt.get_cmap('Blues', 10), plt.get_cmap('Greens', 10)]
    for i, d in enumerate([d1, d2, d3, d4]):
        acc = d['test_accuracies']['test']
        en = d['energies_l1']
        for j, (a, e) in enumerate(zip(acc.T, en.T)):
            plt.plot(a[40:], e[40:], color=pal[i](j), alpha=0.5)
    plt.yscale('log')
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
    
def v2_test():
#if __name__ == '__main__':
    #data = []
    #for DEGREES in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 5
    BATCH_SIZE = 1
    N_IN = 784
    N_HID = 60# np.linspace(20, 100, N_NETWORKS, dtype=int)
    N_OUT = 10
    DEGREES = np.logspace(-1, 1.5, N_NETWORKS-1)
    DEGREES = np.append(2, DEGREES)
    LR = 0.01 #np.logspace(-5, -3, N_NETWORKS)
    margins = np.linspace(0.0, 2.0, N_NETWORKS)
    EPOCHS = 0.05
    TEST_EVERY = 0.025#%

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='datasets',
                                split='digits', # used as a quick swap for EMNIST
                                train=True,
                                download=True,
                                transform=transform, # automatically coverts 0 - 255 --> 0 - 1
                                target_transform=dm.temp_onehot)
    
    test_dataset = datasets.MNIST(root='datasets',
                                split='digits', # used as a quick swap for EMNIST
                                train=False,
                                download=True,
                                transform=transform, # automatically coverts 0 - 255 --> 0 - 1
                                target_transform=dm.temp_onehot)
    
    train = dm.DatasetWithIdx(train_dataset, task='classify')
    test = dm.DatasetWithIdx(test_dataset, task='classify')
    
    # TODO: sampler that takes idxs and batches them dynamically
    bpcounter = interceptors.PerSampleBackwardCounter(240_000)
    s=samplers.RandomSampler(train, N_NETWORKS, 1)
    # s=samplers.VaryBatchAndDatasetSizeSampler(train, 
    #                                          N_NETWORKS, 
    #                                          np.linspace(5_000, 30_000, N_NETWORKS),
    #                                          np.linspace(1, 16, N_NETWORKS, dtype=int),
    #                                          padding_value = -1, 
    #                                          drop_last = True,
    #                                          order = 'identical', # or 'random'
    #                                          method = 'loop') # or 'loop' or 'buffer'
    # test_sampler = samplers.FixedEpochSampler(train,
    #                                s.get_samples_per_network(),
    #                                batch_size=32)
    general_collate = samplers.collate_fn(N_NETWORKS)
    
    #samples_used = s.get_samples_per_network()

    train_dataloader = DataLoader(train,
                                  #batch_size=1)
                              pin_memory=True,
                              num_workers=4,
                              batch_sampler=s,
                              collate_fn=general_collate)
    
    eval_dataloader = DataLoader(train,
                                 batch_size=64,
                                 num_workers=4,
                                 shuffle=False)
    print(len(train_dataloader))
    test_dataloader = DataLoader(test,
                              num_workers=4,
                              batch_size=2048,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(784, 0.5),
                                                                     'b' :1/np.power(784, 0.5)}
                                                        ),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(N_HID, 0.5),
                                                                     'b' : 1/np.power(N_HID, 0.5)}
                                                        ),
                          ).to(DEVICE)

    optimizer = batch_optimizers.AdamW(model.parameters(), lr=0.0005)
    
    # batch_optimizers_temp.Competitive(
    #     batch_optimizers_temp.SGD(model.parameters(), lr=0.01, momentum=0.0), 
    #     k=np.array([np.logspace(-4, 0, N_NETWORKS), # weight
    #                 np.zeros(N_NETWORKS)+0.0, # bias 
    #                 np.zeros(N_NETWORKS)+0.1, # weight
    #                 np.zeros(N_NETWORKS)+0.0]), # bias
    #                 selection_key='grad',
    #                 competition_mode='neuron_wise_weight',
    #                 neuron_competition_dim='incoming',
    #                 bias_competition=True)
    # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    #                                                         reduction='mean'),
    #                                    per_sample=True,
    #                                    reduction='mean') # note batch_losses
    
    #criterion1 = batch_losses.MSELoss(reduction='mean')
    # criterion1 = batch_losses.MAELoss(reduction='mean')
    criterion1 = batch_losses.HingeLoss(margin=0.10)
    #criterion1 = batch_losses.CrossEntropyLoss(reduction='mean')
    #criterion = batch_losses.LazyLoss(batch_losses.MSELoss(reduction='mean'),
    #                                   reduction='mean')
    # criterion1 = batch_losses.StatefulLazyLoss(batch_losses.CrossEntropyLoss(reduction='mean'),
    #                                   max_samples=60_000,
    #                                   n_networks=N_NETWORKS,
    #                                   reduction='mean')

    previous_param_provider = interceptors.PreviousParameterProvider()
    initial_param_provider = interceptors.InitialParameterProvider()
    #prev_prev_param_provider = interceptors.PreviousPreviousParameterProvider(previous_param_provider)
    
    #(mask1, mask2) = masks.create_subnetwork_mask(N_NETWORKS, [784, 100, 10], [np.linspace(30, 90, N_NETWORKS, dtype=int)])
#     mask1 = (masks.MaskComposer(n_linears=N_NETWORKS, n_out=N_HID, n_in=N_IN)
#         .start_with(
#             masks.create_local_connectivity_mask, 
#             image_w=28, image_h=28, kernel_sizes=7
#         )
#         .union(
#             masks.MaskComposer(n_linears=N_NETWORKS, n_out=N_HID, n_in=N_IN)  # <-- IMPORTANT: New instance for the inner chain
#             .start_with(
#                 masks.create_neuron_selection_mask, 
#                 density=0.1
#             )
#             .intersect(
#                 masks.create_local_connectivity_mask, 
#                 image_w=28, image_h=28, kernel_sizes=7
#             )
#         )
#     .get_mask(mask_activities=True, mask_gradients=True)
# )

    # mask1 = (masks.MaskComposer(n_linears=N_NETWORKS, n_in=N_IN, n_out=N_HID)
    #                 .start_with(
    #                     masks.create_local_connectivity_mask, 
    #                     image_w=28,
    #                     image_h=28,
    #                     kernel_sizes=7
    #                 )
    #              ).get_mask(mask_activities=False, mask_gradients=True)

    # mask1 = (masks.MaskComposer(n_linears=N_NETWORKS, n_in=N_IN, n_out=N_HID)
    #             .start_with(
    #                 masks.create_neuron_selection_mask, 
    #                 density=0.0,
    #             )
    #          ).get_mask(mask_activities=False, mask_gradients=True)
    # mask1 = mc.start_with(masks.create_random_mask(N_NETWORKS, N_IN, N_HID, density=0.062)
    #                       ).union(masks.create_random_neuron_mask, density=0.01).get_config()
    #mask2 = masks.create_random_mask(N_NETWORKS, N_HID, N_OUT)
    
    #handler1 = interceptors.ParameterDeltaHandler(2.0, previous_param_provider, mode='cumulative', granularity='network', components=['total', 'weight', 'bias'])
    #handler2 = interceptors.ParameterDeltaHandler(2.0, previous_param_provider, mode='cumulative', granularity='layerwise', components=['total', 'weight', 'bias'])
    #handler3 = interceptors.ParameterDeltaHandler(2.0, initial_param_provider, mode='displacement', granularity='network', components=['total', 'weight', 'bias'])
    #handler4 = interceptors.ParameterDeltaHandler(2.0, initial_param_provider, mode='displacement', granularity='layerwise', components=['total', 'weight', 'bias'])
    #handler5 = interceptors.ParameterDeltaHandler(1.0, previous_param_provider, mode='cumulative', granularity='network', components=['total', 'weight', 'bias'])
    #handler6 = interceptors.ParameterDeltaHandler(2.5, initial_param_provider, mode='displacement', granularity='layerwise')

    #param_iterator = interceptors.ParameterIterator(handlers=[handler1, handler2])
    
    mask1 = masks.create_vary_width_masks(N_NETWORKS, N_IN, np.arange(10, 310, N_NETWORKS))
    mask2 = masks.create_vary_width_masks(N_NETWORKS, np.arange(10, 310, N_NETWORKS), N_OUT)
    trackers = [interceptors.Timer(),
                #interceptors.MaskLinear(model[0], mask1),
                #interceptors.MaskLinear(model[1], mask2),
                #interceptors.TestingLossTracker({'test': ['MSELoss']}),
                #interceptors.TestingAccuracyTracker(['test']),
                interceptors.EpochCounter(240_000),
                previous_param_provider,
                initial_param_provider,
                # interceptors.ParameterDeltaTracker(0.5, previous_param_provider, mode='cumulative', granularity='network', components=['total', 'weight', 'bias']),
                # interceptors.ParameterDeltaTracker(0.75, initial_param_provider, mode='displacement', granularity='layerwise'),
                # interceptors.ParameterDeltaTracker(1.0, previous_param_provider, mode='cumulative', granularity='network', components=['total', 'weight', 'bias']),
                # interceptors.ParameterDeltaTracker(1.5, initial_param_provider, mode='displacement', granularity='layerwise'),
                interceptors.EnergyMetricTracker(1.0, previous_param_provider, mode='energy', granularity='network', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, previous_param_provider, mode='energy', granularity='layerwise', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, previous_param_provider, mode='energy', granularity='neuronwise', components=['weight', 'bias'], energy_direction=['outgoing', 'incoming']),
                interceptors.EnergyMetricTracker(1.0, initial_param_provider, mode='minimum_energy', granularity='network', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, initial_param_provider, mode='minimum_energy', granularity='layerwise', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, initial_param_provider, mode='minimum_energy', granularity='neuronwise', components=['weight', 'bias'], energy_direction=['outgoing', 'incoming']),
                interceptors.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : criterion1},
                                   track_accuracy=True),
                # interceptors.TestLoop('train', 
                #                    eval_dataloader, 
                #                    criterions={'MSELoss' : criterion1},
                #                    track_accuracy=True),
                interceptors.BackwardPassCounter(),
                bpcounter,
                #interceptors.MistakeReplay(train, optimizer, replay_frequency=200, n_replays=np.linspace(0, 10, N_NETWORKS, dtype=int), batch_size=32),
                #interceptors.EnergyL1NetworkTracker(previous_param_provider),
                #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                #interceptors.EnergyL1LayerwiseTracker(previous_param_provider),
                #interceptors.EnergyL0NetworkTracker(previous_param_provider),
                #interceptors.MinimumEnergyL0NetworkTracker(initial_param_provider),
                interceptors.ResultPrinter({'time_taken' : True, 
                                            'test_accuracies' : True,
                                            'test_losses' : ['MSELoss'],
                                            'energies_l1.0_network' : ['total'],
                                            'energies_l1.0_neuronwise' : ['total'],
                                            'func' : {'mean' : np.mean}})]
    
    start=time.time()
    
    #print(list(model.parameters()))
    
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer.train_loop(0.001, 0.001)
    
    d3=utils.convert_data(trainer.state['data'])
    utils.print_data_structure(d3, 'd3')
    trainer.cleanup()
    
    
    
    # model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
    #                                                     activation=nn.GELU(),
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -1/np.power(784, 0.5),
    #                                                                  'b' :1/np.power(784, 0.5)}
    #                                                     ),
    #                       # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
    #                       #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    #                       #                               mu_lr=np.linspace(0.01, 0.1, 5)),
    #                       trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -1/np.power(100, 0.5),
    #                                                                  'b' : 1/np.power(100, 0.5)}
    #                                                     ),
    #                       ).to(DEVICE)
    
    # optimizer = batch_optimizers.AdamW(model.parameters(), lr=LR, beta1=LR)
    # # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    # #                                                         reduction='mean'),
    # #                                    per_sample=True,
    # #                                    reduction='mean') # note batch_losses
    
    # criterion1 = batch_losses.HingeLoss(per_sample=True, 
    #                                     reduction='mean',
    #                                     func=nn.Softplus(beta=3.0),
    #                                     margin=margins)

    # previous_param_provider = interceptors.PreviousParameterProvider()
    # initial_param_provider = interceptors.InitialParameterProvider()
    # prev_prev_param_provider = interceptors.PreviousPreviousParameterProvider(previous_param_provider)
    
    
    # handlers = [interceptors.EnergyL0NetworkHandler()]
    # trackers = [interceptors.Timer(),
    #             interceptors.TestingLossTracker({'test': ['MSELoss']}),
    #             interceptors.TestingAccuracyTracker(['test']),
    #             previous_param_provider,
    #             initial_param_provider,
    #             interceptors.TestLoop('test', 
    #                                test_dataloader, 
    #                                criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
    #                                                                             reduction='sum')}),
    #             interceptors.BackwardPassCounter(),
    #             interceptors.EnergyL0NetworkTracker(),
    #             interceptors.EnergyL1NetworkTracker(previous_param_provider),
    #             interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
    #             #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
    #             #interceptors.ParameterIterator(handlers)]
    
    # s=time.time()
    
    # #print(list(model.parameters()))
    
    # trainer = trainers.Trainer(model, 
    #                            N_NETWORKS, 
    #                            optimizer, 
    #                            criterion1, 
    #                            train_dataloader, 
    #                            test_dataloader, 
    #                            trackers=trackers, 
    #                            device=DEVICE)
    # trainer.train_loop(EPOCHS, TEST_EVERY, sample_increment=1)
    
    # d2=utils.convert_data(trainer.state['data'])
    
    # model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
    #                                                     activation=nn.GELU(),
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -1/np.power(784, 0.5),
    #                                                                  'b' :1/np.power(784, 0.5)}
    #                                                     ),
    #                       # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
    #                       #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    #                       #                               mu_lr=np.linspace(0.01, 0.1, 5)),
    #                       trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -1/np.power(100, 0.5),
    #                                                                  'b' : 1/np.power(100, 0.5)}
    #                                                     ),
    #                       ).to(DEVICE)
    
    # optimizer = batch_optimizers.AdamP(model.parameters(), lr=LR, beta1=0.00000001, beta2=LR)
    # # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    # #                                                         reduction='mean'),
    # #                                    per_sample=True,
    # #                                    reduction='mean') # note batch_losses
    
    # criterion1 = batch_losses.StatefulLazyLoss(batch_losses.HingeLoss(per_sample=True, 
    #                                                                   reduction='mean',
    #                                                                   func=nn.Softplus(beta=3.0),
    #                                                                   margin=margins),
    #                                         max_samples=60_000,
    #                                         n_networks=N_NETWORKS)

    # previous_param_provider = interceptors.PreviousParameterProvider()
    # initial_param_provider = interceptors.InitialParameterProvider()
    # prev_prev_param_provider = interceptors.PreviousPreviousParameterProvider(previous_param_provider)
    
    
    # handlers = [interceptors.EnergyL0NetworkHandler()]
    # trackers = [interceptors.Timer(),
    #             interceptors.TestingLossTracker({'test': ['MSELoss']}),
    #             interceptors.TestingAccuracyTracker(['test']),
    #             previous_param_provider,
    #             initial_param_provider,
    #             interceptors.TestLoop('test', 
    #                                test_dataloader, 
    #                                criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
    #                                                                             reduction='sum')}),
    #             interceptors.BackwardPassCounter(),
    #             interceptors.EnergyL0NetworkTracker(),
    #             interceptors.EnergyL1NetworkTracker(previous_param_provider),
    #             interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
    #             #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
    #             #interceptors.ParameterIterator(handlers)]
    
    # s=time.time()
    
    # #print(list(model.parameters()))
    
    # trainer = trainers.Trainer(model, 
    #                            N_NETWORKS, 
    #                            optimizer, 
    #                            criterion1, 
    #                            train_dataloader, 
    #                            test_dataloader, 
    #                            trackers=trackers, 
    #                            device=DEVICE)
    # trainer.train_loop(EPOCHS, TEST_EVERY, sample_increment=1)
    
    # d3=utils.convert_data(trainer.state['data'])
    
    # model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
    #                                                     activation=nn.GELU(),
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -1/np.power(784, 0.5),
    #                                                                  'b' :1/np.power(784, 0.5)}
    #                                                     ),
    #                       # trainables.BatchDecorrelation(N_NETWORKS, N_HID, 
    #                       #                               decor_lr=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    #                       #                               mu_lr=np.linspace(0.01, 0.1, 5)),
    #                       trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
    #                                                     init_method='uniform',
    #                                                     init_config={'a' : -1/np.power(100, 0.5),
    #                                                                  'b' : 1/np.power(100, 0.5)}
    #                                                     ),
    #                       ).to(DEVICE)
    
    # optimizer = batch_optimizers.LazyAdamW(model.parameters(), lr=LR)
    # # criterion1 = batch_losses.LazyLoss(batch_losses.MSELoss(per_sample=True,
    # #                                                         reduction='mean'),
    # #                                    per_sample=True,
    # #                                    reduction='mean') # note batch_losses
    
    # criterion1 = batch_losses.LazyLoss(batch_losses.HingeLoss(per_sample=True, 
    #                                                                   reduction='mean',
    #                                                                   func=nn.Softplus(beta=3.0),
    #                                                                   margin=margins))

    # previous_param_provider = interceptors.PreviousParameterProvider()
    # initial_param_provider = interceptors.InitialParameterProvider()
    # prev_prev_param_provider = interceptors.PreviousPreviousParameterProvider(previous_param_provider)
    
    
    # handlers = [interceptors.EnergyL0NetworkHandler()]
    # trackers = [interceptors.Timer(),
    #             interceptors.TestingLossTracker({'test': ['MSELoss']}),
    #             interceptors.TestingAccuracyTracker(['test']),
    #             previous_param_provider,
    #             initial_param_provider,
    #             interceptors.TestLoop('test', 
    #                                test_dataloader, 
    #                                criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, 
    #                                                                             reduction='sum')}),
    #             interceptors.BackwardPassCounter(),
    #             interceptors.EnergyL0NetworkTracker(),
    #             interceptors.EnergyL1NetworkTracker(previous_param_provider),
    #             interceptors.EnergyL1LayerwiseTracker(previous_param_provider)]
    #             #interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
    #             #interceptors.ParameterIterator(handlers)]
    
    # s=time.time()
    
    # #print(list(model.parameters()))
    
    # trainer = trainers.Trainer(model, 
    #                            N_NETWORKS, 
    #                            optimizer, 
    #                            criterion1, 
    #                            train_dataloader, 
    #                            test_dataloader, 
    #                            trackers=trackers, 
    #                            device=DEVICE)
    # trainer.train_loop(EPOCHS, TEST_EVERY, sample_increment=1)
    
    # d4=utils.convert_data(trainer.state['data'])
    
    # pal = [plt.get_cmap('Greys', 10), plt.get_cmap('Reds', 10), plt.get_cmap('Blues', 10), plt.get_cmap('Greens', 10)]
    # for i, d in enumerate([d1, d2, d3, d4]):
    #     acc = d['test_accuracies']['test']
    #     en = d['energies_l1']
    #     for j, (a, e) in enumerate(zip(acc.T, en.T)):
    #         plt.plot(a, e, color=pal[i](j), alpha=0.5)
    # plt.yscale('log')
    
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

#def test_two():
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 10
    BATCH_SIZE = 1
    N_IN = 784
    N_HID = 100
    N_OUT = 10
    LR = 0.001
    N_EPOCHS = 10

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
    
    s=samplers.RandomSampler(train, N_NETWORKS, 1)
    general_collate = samplers.collate_fn(N_NETWORKS)

    train_dataloader = DataLoader(train,
                              pin_memory=True,
                              num_workers=0,
                              batch_sampler=s,
                              collate_fn=general_collate)
    
    eval_dataloader = DataLoader(train,
                                 batch_size=16,
                                 num_workers=4,
                                 shuffle=False)

    test_dataloader = DataLoader(test,
                              batch_size=16,
                              num_workers=0,
                              shuffle=False)
    
    model = nn.Sequential(trainables.BatchLinear(N_NETWORKS, N_IN, N_HID, 
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(784, 0.5),
                                                                     'b' :1/np.power(784, 0.5)}
                                                        ),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_HID,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_HID,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          trainables.BatchLinear(N_NETWORKS, N_HID, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.power(100, 0.5),
                                                                     'b' : 1/np.power(100, 0.5)}
                                                        ),
                          ).to(DEVICE)
    
    optimizer = batch_optimizers.SGD(model.parameters(), lr=LR)
    replay_optimizer = batch_optimizers.SGD(model.parameters(), lr=0.01*16)
    
    replay_criterion = batch_losses.CrossEntropyLoss()
    criterion = batch_losses.HingeLoss(margin=0.1)
    
    previous_param_provider = interceptors.PreviousParameterProvider()
    initial_param_provider = interceptors.InitialParameterProvider()
    trackers = [interceptors.Timer(),
                interceptors.EpochCounter(60_000),
                interceptors.RememberSamples(60_000),
                previous_param_provider,
                initial_param_provider,
                interceptors.EnergyMetricTracker(1.0, previous_param_provider, mode='energy', granularity='network', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, previous_param_provider, mode='energy', granularity='layerwise', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, previous_param_provider, mode='energy', granularity='neuronwise', components=['weight', 'bias'], energy_direction=['outgoing', 'incoming']),
                interceptors.EnergyMetricTracker(1.0, initial_param_provider, mode='minimum_energy', granularity='network', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, initial_param_provider, mode='minimum_energy', granularity='layerwise', components=['total', 'weight', 'bias']),
                interceptors.EnergyMetricTracker(1.0, initial_param_provider, mode='minimum_energy', granularity='neuronwise', components=['weight', 'bias'], energy_direction=['outgoing', 'incoming']),
                #interceptors.MaskLinear(model[0], mask),
                #adv0_mask,
                #adv1_mask,
                interceptors.MistakeReplay(train, optimizer, 100, 5, 16, forget=True, criterion=replay_criterion),
                interceptors.TestLoop('test', 
                                   test_dataloader, 
                                   criterions={'MSELoss' : criterion},
                                   track_accuracy=True),
                # interceptors.WeightStatsTracker(['weights', 'gradients'], 
                #                                 stats_to_track=['mean', 'std', 'max', 'min', 'norm'],
                #                                 granularity='global'),
                interceptors.BackwardPassCounter(),
                interceptors.ResultPrinter({'time_taken' : True, 
                                            'test_accuracies' : True,
                                            'test_losses' : ['MSELoss'],
                                            'energies_l1.0_network' : ['total']})]
    trainer2 = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    trainer2.train_loop(0.05, 0.005)
    
    
