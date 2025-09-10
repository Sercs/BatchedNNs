from lib import trainables, trainers, interceptors, batch_losses, batch_optimizers, samplers
from lib import data_manager as dm

from lib.utils import print_state_data_structure

import time

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_NETWORKS = 10
    N_IN = 784
    N_HIDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # demo-ing that we can test multi-sized networks
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
    
    train = dm.DatasetWithIdx(train_dataset, task='classify') # returns idx for tracking
    test = dm.DatasetWithIdx(test_dataset, task='classify')
    
    s=samplers.VaryBatchAndDatasetSizeSampler(
        train, N_NETWORKS, dataset_sizes=np.linspace(10_000, 60_000, N_NETWORKS), 
        batch_sizes=[int(2**i) for i in np.linspace(0, 4, N_NETWORKS)], 
        method='loop', # since small dataset sizes and large batches will hit epoch faster
                       # we can either:
                       #   - loop and waste no resources (but making defining epoch harder on the user)
                       #   - buffer and waste resources on the end (mostly used because it was easy)
                       #   - stretch and distribute wasted resources (makes it better with test intervals)
        order='identical' # ensures networks see the same items and if possible, at the same time
                          # wouldn't happen here due to the different dataset sizes
                          # 'random' gives each network its own dataset of random items
    )
    general_collate = samplers.collate_fn(N_NETWORKS)
    
    samples_used = s.get_samples_per_network()

    train_dataloader = DataLoader(train,
                                  num_workers=4, # 0 > seems slow on cpu-only windows, GPU is much better
                                  batch_sampler=s, # batch_samplers are required
                                  collate_fn=general_collate) # helper function also required
    
    validation_dataloader = DataLoader(train, # note train
                                 batch_size=16,
                                 num_workers=4,
                                 shuffle=False)

    test_dataloader = DataLoader(test,
                              batch_size=16,
                              num_workers=4,
                              shuffle=False)
                             # BatchLinear is the default BatchLinearMasked implements multi-sized networks
    model = nn.Sequential(trainables.BatchLinearMasked(N_NETWORKS, N_IN, N_HIDS, # note that n_hids is a list
                                                        activation=nn.GELU(),
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(784),
                                                                     'b' : 1/np.sqrt(784)
                                                                     }).to(DEVICE), # send to device
                          trainables.BatchLinearMasked(N_NETWORKS, N_HIDS, N_OUT,
                                                        init_method='uniform',
                                                        init_config={'a' : -1/np.sqrt(np.array(N_HIDS)), # note we can also use a list in init
                                                                     'b' : 1/np.sqrt(np.array(N_HIDS))
                                                                     }).to(DEVICE)
                          )
             # batch_optimizers implements SGD-M and Adam(W) for batched hyperparameters
    optimizer = batch_optimizers.AdamW(model.parameters(), lr=np.linspace(0.0001, 0.001, N_NETWORKS),
                                                           beta1=np.linspace(0.5, 0.9, N_NETWORKS),
                                                           device=DEVICE)
    criterion1 = batch_losses.MSELoss(per_sample=True, reduction='mean')
    
    previous_param_provider = interceptors.PreviousParameterProvider() # provides previous parameters for energy calcs
    initial_param_provider = interceptors.InitialParameterProvider() # provides initial parameters for energy calcs
    
    # slight hack to get around the fact we need previous parameters for energy calculations but this 
    # happens before we calculate gradients and actually step, if we want to regularize against 
    # previous parameters we need to go one step further back
    prev_prev_param_provider = interceptors.PreviousPreviousParameterProvider(previous_param_provider)
    
    # handlers are managed by an interceptor that loops through layers in a model and calls them for
    # each parameter group.
    # otherwise each interceptor that computes over parameters like EnergyL1NetworkTracker loops over
    # layers for each interceptor. 
    
    # for interceptor in interceptors                    for layer in layers
    #     for layer in interceptor.layers      vs.           interceptor1_func(layer)
    #         func(layer)                                    interceptor2_func(layer)
    
    
                                                            # which direction should we consider the energy?
                              # note 'neuronwise'           #                        |
    handlers = [interceptors.EnergyL2NeuronwiseHandler(previous_param_provider, ['incoming']), # why not both?
                interceptors.MinimumEnergyL2NeuronwiseHandler(initial_param_provider, ['incoming', 'outgoing']),
                interceptors.EnergyL1LayerwiseHandler(previous_param_provider), # L1 layerwise energies (i.e. 0.weight, 0.bias, 1.weight,...)
                interceptors.MinimumEnergyL1LayerwiseHandler(initial_param_provider), # note that bias is considered separately
                interceptors.EnergyL0NetworkHandler(), # L0 doesn't need a provider as it looks at grads directly
                interceptors.MinimumEnergyL0NetworkHandler(initial_param_provider)] # L0s are explicitly calculated (not estimated)
    
    trackers = [interceptors.Timer(), # tracks time (trainer currently prints time so this is required)
                interceptors.RunningLossTracker(), # tracks running loss (which is divided by the batch_size used)
                                                   # and therefore hard to compare with test loops.
                interceptors.RunningAccuracyTracker(), # tracks accuracy (not divided though)
                                      #      name of test loops       name of criterion(s) used in test loop(s)
                                      #            |                         |          |
                interceptors.TestingLossTracker({'test' : ['MSELoss', 'LazyLoss'],#     |
                                                 'validation' : ['LazyLoss', 'CrossEntropyLoss']}), 
                interceptors.TestingAccuracyTracker(['test']),
                interceptors.TestLoop('test', # <- name of test loop used above
                                   test_dataloader, # note we use the test set
                                                                                  # per_sample is used for training.
                                                                                  # sum used since we want average 
                                      #  name of criterion used above             # sample and this may change with
                                      #           |                               # batch size
                                   criterions={'MSELoss' : batch_losses.MSELoss(per_sample=False, reduction='sum'),
                                               # note that LazyLoss wraps around MSELoss (we could equally use CrossEntropy)
                                               'LazyLoss' : batch_losses.LazyLoss(
                                                   batch_losses.MSELoss(per_sample=True, reduction='sum'),
                                                   per_sample=False,
                                                   reduction='sum')}, 
                                   device=DEVICE),
                # we can run a different test loop for training accuracy (though I need to work on the smaller dataset case)
                interceptors.TestLoop('validation',
                                   validation_dataloader,
                                   criterions={'CrossEntropyLoss' : batch_losses.CrossEntropyLoss(per_sample=False, reduction='sum'),
                                               'LazyLoss' : batch_losses.LazyLoss( # even though lazyloss overlaps with above we know which test loss its associated with 
                                                   batch_losses.CrossEntropyLoss(per_sample=True, reduction='sum'),
                                                   per_sample=False,
                                                   reduction='sum')}, 
                                   device=DEVICE),
                interceptors.ForwardPassCounter(), # tracks forward passes (i.e. batch=1 -> 1, batch=256 -> 1)
                interceptors.ForwardItemCounter(), # tracks samples in a forward (i.e. batch=1 -> 1, batch=256 -> 256)
                interceptors.BackwardPassCounter(), # counts backward passes even if items in the batch are skipped
                interceptors.BackwardItemCounter(), # considers only the samples used in a batch (useful for lazy)
                interceptors.PerSampleBackwardCounter(max_samples=60_000), # needs to know idxs of samples.
                                                                           # our dataset might only be 10 samples
                                                                           # but if these samples come from a 
                                                                           # 60_000 dataset we need to know those 
                                                                           # which of those 60_000 idxs the 10 are.
                prev_prev_param_provider, # order matters here since this relies on previous_param_provider
                previous_param_provider, # used by energy interceptors and handlers
                initial_param_provider,                                   # if we used previous parameters, we'd get zeros
                                                                          #                        |
                interceptors.L1Regularizer(np.logspace(np.log10(1e-6), np.log10(1e-2), N_NETWORKS), prev_prev_param_provider),
                
                                                        # we can simultaneuosly regularize L2 norm against the initial condition (never tried this but just demo-ing)
                                                        #                                      |
                interceptors.L2Regularizer(np.logspace(np.log10(1e-2), np.log10(1e-6), N_NETWORKS), initial_param_provider), # this improves E_min but doesn't help E
                interceptors.EnergyL1NetworkTracker(previous_param_provider), # we have handlers of these too
                interceptors.MinimumEnergyL1NetworkTracker(initial_param_provider),
                interceptors.ParameterIterator(handlers=handlers)] # note that this takes a list of handlers
    
    s=time.time()

    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=trackers, 
                               device=DEVICE)
    #         when used mixed dataset sizes, we need to know how often to test, so we use manual references
    # train for 5% total epoch and test every 1%       |
    #                      |                           |
    trainer.train_loop(0.05, 0.01, dataset_size=60_000, sample_increment=1)
    for i in range(len(trainer.state['data']['minimum_energies_l0'][0])):
        print(trainer.state['data']['minimum_energies_l0'][0][i] == N_IN*N_HIDS[i] + N_HIDS[i]*N_OUT + N_HIDS[i] + N_OUT)
    print("Time taken with (many) trackers:", time.time()-s)
    print("Data saved:")
    print_state_data_structure(trainer.state['data']) # get the structure of the data we collected (dict and keys)
    # optionally save
    # writes json file with trainer.state['data'] and optional experimental setup
    # trainer.save_data_as_json('./demo', experimental_setup={'initial_conditions' : {'layer 1' : np.sqrt(784),
    #                                                                                 'layer 2' : 1/np.sqrt(np.array(N_HIDS))},
    #                                                         'n_hids' : N_HIDS,
    #                                                         'dataset_sizes' : np.linspace(10_000, 60_000, N_NETWORKS),
    #                                                         'batch_sizes' : [int(2**i) for i in np.linspace(0, 4, N_NETWORKS)],
    #                                                         'L1reg' : np.logspace(np.log10(1e-6), np.log10(1e-2), N_NETWORKS),
    #                                                         'L2reg' : np.logspace(np.log10(1e-2), np.log10(1e-6), N_NETWORKS),
    #                                                         'train method' : 'loop'}
    print("#############################################")
    s = time.time()
    trainer = trainers.Trainer(model, 
                               N_NETWORKS, 
                               optimizer, 
                               criterion1, 
                               train_dataloader, 
                               test_dataloader, 
                               trackers=[interceptors.Timer()], # or we don't have to include anything 
                               device=DEVICE)
    #         when used mixed dataset sizes, we need to know how often to test, so we use manual references
    # train for 5% total epoch and test every 1%       |
    #                      |                           |
    trainer.train_loop(0.05, 0.01, dataset_size=60_000, sample_increment=1)
    print("Time taken with no trackers:", time.time()-s)
    print("Data saved:")
    print_state_data_structure(trainer.state['data'])
