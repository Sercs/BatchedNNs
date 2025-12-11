# TODO: sleep epochs
# TODO: Generalize energy measures, better calculated metrics
# TODO [minor]: weight masking, providers as handlers, std

from .trainables import BatchLinear

from .samplers import FixedEpochSampler as FixedEpochSampler # used in mistake replay
from .samplers import collate_fn as collate_fn # used in mistake replay
from torch.utils.data import DataLoader # used in mistake replay

from collections import deque

import torch
import torch.nn as nn
import numpy as np
import math
import time
"""
    PyTorch-Lightning inspired.
    Interceptors represent any code you'd want to run during training but not
    necessarily during every training run you implement. They listen for 
    functions called in the main training loop to make modifications or 
    logging.
"""
class Interceptor:
    def __init__(self):
        # this will be populated by the Trainer upon initialization.
        self.trainer = None
    # main function listeners
    def before_train(self, state): pass # called before the first epoch, useful for initialisation
    def after_train(self, state): pass # called after all epochs, useful for collating records
    def before_epoch(self, state): pass # called once at the start of an epoch
    def after_epoch(self, state): pass # called at the end of an epoch
    def before_test(self, state): pass # we test intermittently, this runs before that test
    def on_test_run(self, state): pass # this is effectively used in conjunction with a TestLoop interceptor
    def after_test(self, state): pass # runs after the test, useful for updating current metrics
    def before_train_forward(self, state): pass # runs before a train forward pass, where state['x'], state['y'] and state['idx'] will be free to manipulate
    def after_train_forward(self, state): pass # runs after the train forward when the state['loss'] is available to add for instance regularization
    def before_test_forward(self, state): pass # runs before a test forward (where no updating should occur), distinguishing it from a forward pass 
    def after_test_forward(self, state): pass # runs after a test forward
    def before_update(self, state): pass # run before calling into the before the train forward, it means we can make an interceptor run on_update and internally run on_forward_train without it running recursively
    def after_update(self, state): pass # runs after the whole forward, backward, step, apply update pass
    def before_step(self, state): pass # runs just before .step() when grads are available
    def after_step(self, state): pass # runs just after .step(), useful for computing weight changes
    
"""
    Handlers behave like Interceptors, listening for functions to do something.
    However, they only operate on each parameter as it comes. They are 
    typically paired with the ParamIterator(Interceptor) to make looping over 
    layers more efficient.
"""

class Handler:
    # >>> Functions <<<
    def before_train_func(self, name, parameter, state=None): pass
    def after_train_func(self, name, parameter, state=None): pass
    def before_epoch_func(self, name, parameter, state=None): pass
    def after_epoch_func(self, name, parameter, state=None): pass
    def before_test_func(self, name, parameter, state=None): pass
    def after_test_func(self, name, parameter, state=None): pass
    def before_train_forward_func(self, name, parameter, state=None): pass
    def after_train_forward_func(self, name, parameter, state=None): pass
    def before_test_forward_func(self, name, parameter, state=None): pass
    def after_test_forward_func(self, name, parameter, state=None): pass
    def before_update_func(self, name, parameter, state=None): pass
    def after_update_func(self, name, parameter, state=None): pass
    def before_step_func(self, name, parameter, state=None): pass
    def after_step_func(self, name, parameter, state=None): pass
    # >>> Logging <<<
    def before_train_log(self, state=None): pass
    def after_train_log(self, state=None): pass
    def before_epoch_log(self, state=None): pass
    def after_epoch_log(self, state=None): pass
    def before_test_log(self, state=None): pass
    def after_test_log(self, state=None): pass
    def before_train_forward_log(self, state=None): pass
    def after_train_forward_log(self, state=None): pass
    def before_test_forward_log(self, state=None): pass
    def after_test_forward_log(self, state=None): pass
    def before_update_log(self, state=None): pass
    def after_update_log(self, state=None): pass
    def before_step_log(self, state=None): pass
    def after_step_log(self, state=None): pass

class HomeostaticSleepReplay(Interceptor):
    def __init__(self, sample_memorizer, 
                       data_source, 
                       optimizer, 
                       growth_threshold=0.2, 
                       min_samples_to_trigger=10, 
                       sleep_epochs=1,       
                       batch_size=32, 
                       criterion=None, 
                       forget=False, 
                       num_workers=0):
        super().__init__()
        self.sample_memorizer = sample_memorizer
        self.data_source = data_source
        self.optimizer = optimizer
        
        self.growth_threshold = growth_threshold
        self.min_samples_to_trigger = min_samples_to_trigger
        self.sleep_epochs = sleep_epochs 
        self.batch_size = batch_size
        self.criterion = criterion
        self.forget = forget
        self.num_workers = num_workers 
        
    def before_train(self, state):
        n_networks = state['n_networks']
        device = state['device']
        
        # 1. Handle Growth Threshold (Scalar -> Vector)
        if isinstance(self.growth_threshold, (int, float)):
            gt_list = [float(self.growth_threshold)] * n_networks
        else:
            gt_list = self.growth_threshold
        self.growth_threshold_tensor = torch.as_tensor(gt_list, device=device, dtype=torch.float32)
        
        # 2. Handle Sleep Epochs (Scalar -> Vector)
        if isinstance(self.sleep_epochs, (int, float)):
             se_list = [int(self.sleep_epochs)] * n_networks
        else:
             se_list = self.sleep_epochs
        self.sleep_epochs_tensor = torch.as_tensor(se_list, device=device, dtype=torch.long)

        # 3. Initialize Baselines
        self.last_wake_sizes = torch.ones(n_networks, device=device) * self.min_samples_to_trigger
        
        state['data']['sleep_events'] = [] 

    def after_update(self, state):
        if not hasattr(self.sample_memorizer, 'get_samples'):
            return
            
        # Get mask: (Max_Samples, N_Networks) -> Sum -> (N_Networks,)
        current_coreset_sizes = self.sample_memorizer.get_samples().sum(dim=0).float().to(state['device'])
        
        # Calculate Pressure
        pressure = current_coreset_sizes / (self.last_wake_sizes + 1e-9)
        
        # --- LOGIC UPDATE ---
        # 1. Check if pressure exceeds threshold
        trigger_condition = (pressure >= (1.0 + self.growth_threshold_tensor))
        
        # 2. Safety Check: Only trigger if threshold > 0. 
        #    This allows 0.0 to act as a "Disable Replay" (Baseline) setting.
        is_enabled = (self.growth_threshold_tensor > 0.0)
        
        needs_sleep = trigger_condition & is_enabled
        # --------------------

        if not needs_sleep.any():
            return

        max_epochs_needed = self.sleep_epochs_tensor[needs_sleep].max().item()
        criterion = state['criterion'] if self.criterion is None else self.criterion
        
        for epoch_step in range(1, max_epochs_needed + 1):
            
            # Active if: Needs Sleep AND Epoch count is within budget
            active_this_loop = (needs_sleep & (epoch_step <= self.sleep_epochs_tensor)).float()
            
            if active_this_loop.sum() == 0:
                break
            
            per_sample_backward_counts = self.sample_memorizer.get_samples()
            
            idxs_to_replay = [torch.tensor(np.where(n > 0)[0]) if active_this_loop[i] else [] for i, n in enumerate(per_sample_backward_counts.T)]
            
            replay_sampler = FixedEpochSampler(self.data_source,
                                             idxs_to_replay,
                                             self.batch_size,
                                             padding_value=state['padding_value'])
                                             
            replay_dataloader = DataLoader(self.data_source,
                                         batch_sampler=replay_sampler,
                                         collate_fn=collate_fn(state['n_networks']),
                                         num_workers=self.num_workers,
                                         shuffle=False)
            for (x, y, idx) in replay_dataloader:
                x, y, idx = x.to(state['device']), y.to(state['device']), idx.to(state['device']) 
                
                if len(x.shape) < 3:
                    x = x.unsqueeze(1)
                    y = y.unsqueeze(1).repeat((1, state['n_networks'], 1))
                    idx = idx.unsqueeze(1).repeat(1, state['n_networks'])   
                
                state['x'], state['y'], state['idx'] = x, y, idx
                
                self.trainer._fire_event('before_update') 
                self.trainer._fire_event('before_train_forward')
                
                y_hat = state['model'](x)
                self.optimizer.zero_grad()
                
                per_sample_supervised_loss = criterion(y_hat, y, idx, state['padding_value'])
                state['per_sample_losses'] = per_sample_supervised_loss
                active_mask = active_this_loop.unsqueeze(0)
                mask = (idx != state['padding_value'])
                n_valid_samples = mask.sum(0)
                
                supervised_loss = (per_sample_supervised_loss.sum(0) * active_mask) / (n_valid_samples + 1e-12)
                state['loss'] = supervised_loss
                
                self.trainer._fire_event('after_train_forward')
                                      
                supervised_loss.sum().backward()
                
                self.trainer._fire_event('before_step')
                self.optimizer.step()
                self.trainer._fire_event('after_step')
                
                if self.forget:
                     self.sample_memorizer.forget_samples(y_hat, y, idx, state['padding_value'])

        # Update baselines for those that slept
        self.last_wake_sizes[needs_sleep] = current_coreset_sizes[needs_sleep]
        
        state['data']['sleep_events'].append({
            'step': state['step'],
            'slept_mask': needs_sleep.cpu().numpy(),
            'new_baselines': self.last_wake_sizes.cpu().numpy()
        })
        
class ActiveMemorySizeTracker(Interceptor):
    """
    Tracks the 'Active Set' Size (M): The absolute number of unique samples 
    that the model has effectively 'remembered' (trained on).
    
    This corresponds to the numerator of the Scaling Law ratio.
    
    Prerequisites:
        - Must be placed AFTER 'PerSampleBackwardCounter' in the interceptor list.
    """
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        # Initialize the log list
        state['data']['active_memory_size'] = []

    def after_test(self, state):
        counts = state['data'].get('per_sample_backward_counts')
        
        # 1. Fallback if counter isn't present
        if counts is None:
            state['data']['active_memory_size'].append(np.zeros(state['n_networks']))
            return

        # 2. Ensure tensor format
        if isinstance(counts, np.ndarray):
            counts = torch.from_numpy(counts)
            
        # 3. Calculate M (Active Set Size)
        # Sum boolean mask over samples (dim 0) -> Shape: (n_networks,)
        # We perform this on the same device as 'counts' to avoid unnecessary transfers
        active_set_size = (counts > 0).sum(dim=0).float()
        
        # 4. Log
        # Move to CPU/Numpy for storage
        state['data']['active_memory_size'].append(active_set_size.cpu().numpy())

# gemini pro 2.5
class ActiveMemoryFractionTracker(Interceptor):
    """
    Tracks the 'Active Set' fraction (M/N): The proportion of the dataset 
    that the model has effectively 'remembered' (trained on).
    
    This corresponds to the Y-axis of the Scaling Law plot:
        y = (Samples with > 0 updates) / (Total Dataset Size)
        
    Prerequisites:
        - Must be placed AFTER 'PerSampleBackwardCounter' in the interceptor list.
    """
    def __init__(self, dataset_sizes):
        super().__init__()
        # Handle scalar, list, or numpy array inputs for dataset_sizes
        if isinstance(dataset_sizes, (int, float)):
            self.dataset_sizes = torch.as_tensor(dataset_sizes, dtype=torch.float32)
        else:
            self.dataset_sizes = torch.as_tensor(dataset_sizes, dtype=torch.float32)

    def before_train(self, state):
        # Ensure dataset_sizes is on the correct device for tensor operations
        if torch.is_tensor(self.dataset_sizes):
            self.dataset_sizes = self.dataset_sizes.to(state['device'])
            
        state['data']['active_memory_fraction'] = []

    def after_test(self, state):
        counts = state['data'].get('per_sample_backward_counts')
        
        # --- FIX: Fetch the current N_env from the dedicated tracker ---
        unique_seen_list = state['data'].get('unique_samples_seen')
        if not unique_seen_list:
            # If the seen tracker hasn't logged anything, N_env = 1 (or skip)
            N_env = torch.tensor(1.0, dtype=torch.float32).to(self.dataset_sizes.device) 
        else:
            # Get the scalar count (last logged value)
            N_env_scalar = unique_seen_list[-1]
            N_env = torch.tensor(N_env_scalar, dtype=torch.float32).to(self.dataset_sizes.device)
            
        if counts is None:
            # Fallback
            state['data']['active_memory_fraction'].append(np.zeros(state['n_networks']))
            return

        if isinstance(counts, np.ndarray):
            counts = torch.from_numpy(counts)
            
        # 1. Calculate M (Active Set Size)
        # Sum boolean mask over samples (dim 0) 
        active_set_size = (counts > 0).sum(dim=0).float()
        
        # 2. Match devices for division (M_active and N_env are moved to GPU)
        M_active = active_set_size.to(self.dataset_sizes.device)
        
        # 3. Calculate Fraction (M / N_env)
        fraction = M_active / N_env
        
        # 4. Log
        state['data']['active_memory_fraction'].append(fraction.cpu().numpy())

# gemini pro 2.5
class UniqueSampleSeenTracker(Interceptor):
    """
    Tracks the total count of unique global indices seen by *each* network 
    stream independently. Logs the result as a tensor of shape (n_models,).
    """
    def __init__(self, max_samples):
        super().__init__()
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")
        self.max_samples = max_samples

    def before_train(self, state):
        n_networks = state['n_networks']
        # Initialize a persistent 2D boolean mask: (max_samples, n_networks)
        # Row = Global Sample ID; Column = Network ID
        self.seen_mask = torch.zeros((self.max_samples, n_networks), dtype=torch.bool, device='cpu')
        
        # We reuse the original log key, but the data structure now holds per-model counts
        state['data']['unique_samples_seen'] = [] 

    # Listen to after_step to capture all batch indices and update the 2D mask
    def after_step(self, state):
        idx = state.get('idx')
        padding_value = state.get('padding_value', -1)

        if idx is None:
            return

        # Move tensors to CPU to match the persistent mask and ensure safety
        idx_cpu = idx.detach().cpu()
        
        # 1. Create a validity mask (where indices are NOT padding)
        is_valid = (idx_cpu != padding_value)

        # 2. Find the coordinates (batch_row, network_col) for all valid samples
        # coords is a tuple of (row indices, col indices) for the 2D batch tensor
        coords = torch.where(is_valid)

        if coords[0].numel() == 0:
            return

        # 3. Extract Global Sample IDs (The row index for the persistent mask)
        # Uses advanced indexing on the idx tensor to get the actual global ID at that coordinate
        global_sample_ids = idx_cpu[coords].long() 

        # 4. Extract Network IDs (The column index for the persistent mask)
        network_ids = coords[1].long() 

        # 5. Update the 2D mask (Target shape: [max_samples, n_networks])
        # This is the key update: Set the specific (Sample ID, Network ID) cell to True
        self.seen_mask[global_sample_ids, network_ids] = True 

    def after_test(self, state):
        """
        Calculates the count of unique samples seen PER network (summing along dim 0)
        and logs the resulting tensor of shape (n_models,).
        """
        # Sum the mask along the sample dimension (dim 0) to get the count PER network
        # Result shape: (n_networks,)
        unique_count_tensor = self.seen_mask.sum(dim=0).float()
        
        # Log the tensor
        state['data']['unique_samples_seen'].append(unique_count_tensor.numpy())
        
class MistakeReplay(Interceptor):
    def __init__(self, sample_memorizer, 
                       data_source, 
                       optimizer, 
                       replay_frequency, 
                       n_replays, 
                       batch_size, 
                       criterion=None, 
                       forget=False, 
                       num_workers=0):
        super().__init__()
        self.data_source = data_source
        self.optimizer = optimizer
        self.replay_frequency = replay_frequency # TODO: check list
        self.n_replays = n_replays
        self.batch_size = batch_size
        self.forget = forget
        self.criterion = criterion
        self.sample_memorizer = sample_memorizer
        self.num_workers = num_workers 
        
    def before_train(self, state):
        n_networks, device = state['n_networks'], state['device']
        freq = self.replay_frequency if isinstance(self.replay_frequency, (list, np.ndarray, torch.Tensor)) else [self.replay_frequency] * n_networks
        replays = self.n_replays if isinstance(self.n_replays, (list, np.ndarray, torch.Tensor)) else [self.n_replays] * n_networks
        self.replay_frequencies = torch.tensor(freq, device=device)
        self.n_replays = torch.tensor(replays, device=device)
    
    def after_update(self, state):
        step = state['step']
        active_on_freq = (self.replay_frequencies > 0) & (step % self.replay_frequencies == 0)
        if not active_on_freq.any():
            return

        if not hasattr(self.sample_memorizer, 'get_samples'):
            print("⚠️ '.get_samples()' not in sample_memorizer. Skipping replay.")
            return
        
        max_replays_this_step = self.n_replays[active_on_freq].max().item()
        criterion = state['criterion'] if self.criterion is None else self.criterion
        for replay_step in range(1, max_replays_this_step+1): # means n_replays [0, 0, 1, 2] gives no replay to 0s
            per_sample_backward_counts = self.sample_memorizer.get_samples()
            idxs_to_replay = [torch.tensor(np.where(n > 0)[0]) if active_on_freq[i] else [] for i, n in enumerate(per_sample_backward_counts.T)]
            replay_sampler = FixedEpochSampler(self.data_source,
                                           idxs_to_replay,
                                           self.batch_size,
                                           padding_value=state['padding_value'])
            replay_dataloader = DataLoader(self.data_source,
                                         batch_sampler=replay_sampler,
                                         collate_fn=collate_fn(state['n_networks']),
                                         num_workers=self.num_workers,
                                         shuffle=False)
            active_this_loop = (active_on_freq & (replay_step <= self.n_replays)).float()
            # do loop
            for (x, y, idx) in replay_dataloader:
                x, y, idx = x.to(state['device']), y.to(state['device']), idx.to(state['device']) 
                if len(x.shape) < 3:
                    x, y = x.unsqueeze(1), y.unsqueeze(1).repeat((1, state['n_networks'], 1))
                    idx = idx.unsqueeze(1).repeat(1, state['n_networks'])   
                state['x'], state['y'], state['idx'] = x, y, idx
                self.trainer._fire_event('before_update')
                self.trainer._fire_event('before_train_forward')
                y_hat = state['model'](x)
                
                self.optimizer.zero_grad()
                
                per_sample_supervised_loss = criterion(y_hat, y, idx, state['padding_value'])
                state['per_sample_losses'] = per_sample_supervised_loss
                mask = (idx != state['padding_value'])
                n_valid_samples = mask.sum(0)
                supervised_loss = per_sample_supervised_loss.sum(0) * active_this_loop / (n_valid_samples + 1e-12) # average 
                state['loss'] = supervised_loss
                
                correct = ((y_hat.argmax(-1) == y.argmax(-1)) * (idx != state['padding_value'])).sum(0)
                state['correct'] = correct
                
                self.trainer._fire_event('after_train_forward')
                                  
                loss = state['loss']
                
                loss.sum().backward()
                self.trainer._fire_event('before_step')
                self.optimizer.step()
                self.trainer._fire_event('after_step')
                
                if self.forget:
                    self.sample_memorizer.forget_samples(y_hat, y, idx, state['padding_value'])
                    
                # after_update would cause infinite loop, i know because that use to be here
                
class RememberMistakes(Interceptor):
    def __init__(self, max_samples, forget=False):
        super().__init__()
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")
        self.max_samples = max_samples
        self.forget = forget

    def before_train(self, state):
        n_networks = state['n_networks']
        self.remember_samples = torch.zeros((self.max_samples, n_networks), dtype=torch.long)
        state['data']['remember_samples'] = self.remember_samples

    def after_update(self, state):
        per_sample_loss = state.get('per_sample_losses').cpu()
        idx = state.get('idx').cpu()

        if per_sample_loss is None or idx is None:
            return

        # get indices of valid samples that contributed to the loss
        update_coords = torch.where(
            (per_sample_loss > 1e-9) & (idx != state.get('padding_value', -1)) # eps for rounding errors
        )
        
        if update_coords[0].numel() == 0:
            return 

                            #    Gemini-Pro 2.5 tip
                            #             |
                            #             V
        self.remember_samples.index_put_(
            (idx[update_coords], update_coords[1]),
            torch.tensor(1, device='cpu')
        )
        if self.forget:
            self.forget_samples(state['y_hat'], state['y'], state['idx'], state['padding_value'])

    def after_train(self, state):
        state['data']['remember_samples'] = self.remember_samples.clone().numpy()
    
    # interface with replay
    def forget_samples(self, y_hat, y, idx=None, padding_value=None):
        correct = ((y_hat.argmax(-1) == y.argmax(-1)) * (idx != padding_value))
        forget_idxs = torch.where(correct)
        self.remember_samples.index_put_(
            (idx[correct], forget_idxs[1]),
            torch.tensor(0, device=self.remember_samples.device)
            )
        #print(self.remember_samples.sum(0))
    
    # interface with replay
    def get_samples(self):
        return self.remember_samples
    
    # interface with hard mining sampler 
    def get_samples_as_list_per_network(self, i):
        return np.where(self.remember_samples.T[i] > 0)[0].tolist()

############################## TEST LOOP ##################################
"""
An optional test loop. It expects a name which indicates which test is being done
(i.e. test loop, train loop, validation loop,etc) and dictionary of criterions with 
{name of loss function : loss function} so that multiple criterions can be tested
against.
"""



class TestLoop(Interceptor):
    """
    Performs a full evaluation loop, with optional loss and accuracy tracking.
    This class handles its own accumulation and masking for correctness.
    """
    def __init__(self, name, dataloader, criterions=None, track_accuracy=True):
        """
        Args:
            name (str): A unique name for this test context (e.g., 'validation').
            test_dataloader: The DataLoader for the evaluation set.
            criterions (dict, optional): A dictionary of loss functions. If None,
                                         loss is not calculated. Defaults to None.
            track_accuracy (bool, optional): If True, accuracy is calculated.
                                             Defaults to True.
        """
        super().__init__()
        self.name = name
        self.dataloader = dataloader
        self.criterions = criterions or {} # Handle None case
        self.track_accuracy = track_accuracy

    @torch.no_grad()
    def on_test_run(self, state):
        model = state['model']
        device = state['device']
        padding_value = state['padding_value']

        model.eval()

        # --- Initialize accumulators based on configuration ---
        total_loss_sums = {name: 0.0 for name in self.criterions} if self.criterions else {}
        total_correct_sum = torch.zeros(state['n_networks'], device=device) if self.track_accuracy else None
        total_samples = torch.zeros(state['n_networks'], device=device)

        for (x, y, idx) in self.dataloader:
            x, y, idx = x.to(device), y.to(device), idx.to(device)
            if len(x.shape) < 3:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1).repeat((1, state['n_networks'], 1))
                idx = idx.unsqueeze(1).repeat(1, state['n_networks'])
                
            mask = (idx != padding_value).float()

            y_hat = model(x)
            
            if mask.sum() == 0:
                continue

            if self.criterions:
                for name, crit in self.criterions.items():
                    masked_per_sample_loss = crit(y_hat, y, idx, padding_value)
                    total_loss_sums[name] += masked_per_sample_loss.sum(0)

            if self.track_accuracy:
                correct_tensor = (y_hat.argmax(-1) == y.argmax(-1)).float()
                total_correct_sum += (correct_tensor * mask).sum(dim=0)
            
            total_samples += mask.sum(0)

        if self.criterions:
            for name, total_loss in total_loss_sums.items():
                avg_loss = total_loss / (total_samples)
                avg_loss[total_samples == 0] = 0.0
                state['data'].setdefault(f'{self.name}_losses', {}).setdefault(name, []).append(avg_loss.cpu().numpy())

        if self.track_accuracy:
            avg_accuracies = (total_correct_sum / total_samples) * 100
            avg_accuracies[total_samples == 0] = 0.0
            state['data'].setdefault(f'{self.name}_accuracies', []).append(avg_accuracies.cpu().numpy())

        model.train()
        
############################# METRICS ################################
"""
These Interceptors deal with metrics like wall clock speed, losses and accuracies.
"""
# gemini-flash with hand-holding
class ResultPrinter(Interceptor):
    """
    Interceptor that prints specified metrics after every test run, with the
    ability to apply functions like np.mean to array-like results.
    """
    def __init__(self, metrics):
        super().__init__()
        # Separate the metric config from the function config
        self.metrics_config = {k: v for k, v in metrics.items() if k != 'func'}
        self.funcs = metrics.get('func', {})

    def _format_result(self, latest_result):
        """Helper to format a scalar or apply functions to an array."""
        if isinstance(latest_result, (float, int, np.number)):
            return f"{latest_result:.5f}"
        
        elif isinstance(latest_result, (np.ndarray, torch.Tensor)):
            if isinstance(latest_result, torch.Tensor):
                latest_result = latest_result.cpu().numpy()
            
            # If functions are provided, apply them
            if self.funcs:
                results = []
                for func_name, func in self.funcs.items():
                    try:
                        value = func(latest_result)
                        results.append(f"{func_name.title()}: {value:.5f}")
                    except Exception:
                        results.append(f"{func_name.title()}: N/A")
                return ", ".join(results)
            # Otherwise, print the full array (for short arrays)
            else:
                values = [f"{v:.5f}" for v in latest_result.flatten()]
                return f"[{', '.join(values)}]"
                
        return str(latest_result)

    def _print_formatted(self, title, value_str, indent_level=0):
        """Prints title and value with smart formatting."""
        prefix = '  ' * indent_level
        # If the value is complex (multi-part or a full array), print on a new line
        if ',' in value_str or '[' in value_str:
            print(f"{prefix}{title}:")
            print(f"{prefix}  -> {value_str}")
        else: # Simple scalar
            print(f"{prefix}{title}: {value_str}")

    def _recursive_print(self, data, config, indent_level):
        """Recursively traverses and prints nested metric data."""
        targets = []
        if config is True: targets = list(data.keys())
        elif isinstance(config, list): targets = config
        elif isinstance(config, dict): targets = list(config.keys())
        
        for key in targets:
            if key not in data: continue
            
            sub_data = data[key]
            sub_config = config.get(key) if isinstance(config, dict) else True
            
            title = key.replace('_', ' ').title()
            
            # Base Case: We've hit a list of results
            if isinstance(sub_data, list) and sub_data:
                formatted_value = self._format_result(sub_data[-1])
                self._print_formatted(title, formatted_value, indent_level)
            # Recursive Step: Continue deeper
            elif isinstance(sub_data, dict):
                print(f"{'  ' * indent_level}{title}:")
                self._recursive_print(sub_data, sub_config, indent_level + 1)

    def after_test(self, state):
        """Prints the most recent value for each configured metric."""
        if not self.metrics_config: return
        
        print("-" * 50)
        print(f"EPOCH {state.get('epoch', 0)} | RESULTS")
        print("-" * 50)
        
        for metric_name, config in self.metrics_config.items():
            if metric_name not in state['data']:
                print(f"⚠️  Metric '{metric_name}' not found.")
                continue
            
            self._recursive_print(state['data'], {metric_name: config}, 0)

############################# COUNTERS ################################
"""
These Interceptors primarily deal with counting items: forward passes,
the number of items in a forward pass (i.e. batching), the number of 
items actually used for learning, etc.
"""

class Timer(Interceptor):
    def __init__(self):
        super().__init__()
        self.last_test_time = None

    def before_train(self, state):
        state['data']['time_taken'] = []
        self.last_test_time = time.time()

    def after_test(self, state):
        current_time = time.time()
        time_delta = current_time - self.last_test_time
        
        state['data']['time_taken'].append(time_delta)
        
        self.last_test_time = current_time

class EpochCounter(Interceptor):
    def __init__(self, dataset_sizes):
        super().__init__()
        if isinstance(dataset_sizes, (list, np.ndarray, int)):
            dataset_sizes = torch.tensor(dataset_sizes)
        self.dataset_sizes = dataset_sizes
   
    def before_train(self, state):
        self.dataset_sizes = self.dataset_sizes.to(state['device'])
        self.samples_seen = torch.zeros((state['n_networks'],), device=state['device'])
        self.epochs = torch.zeros((state['n_networks'],), device=state['device'])
        state['data']['epochs'] = []
        
    def after_update(self, state):
        sample_count = (state['idx'] != state['padding_value']).sum(0)
        self.samples_seen += sample_count
        self.epochs = self.samples_seen / self.dataset_sizes
    
    def after_test(self, state):
        state['data']['epochs'].append(self.epochs.clone().detach().cpu().numpy())

class ForwardPassCounter(Interceptor):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        n_networks = state['n_networks']
        self.forward_pass_counts = torch.zeros(n_networks, dtype=torch.long)
        state['data']['forward_pass_counts'] = []

    def after_step(self, state):
        idx = state.get('idx')
        padding_value = state.get('padding_value', -1)

        if idx is None:
            return

        is_valid = (idx != padding_value)
        passes_this_batch = is_valid.any(dim=0).long()
        
        self.forward_pass_counts += passes_this_batch.cpu()

    def after_test(self, state):
        state['data']['forward_pass_counts'].append(self.forward_pass_counts.clone().numpy())

class ForwardItemCounter(Interceptor):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        n_networks = state['n_networks']
        self.forward_item_counts = torch.zeros(n_networks, dtype=torch.long)
        state['data']['forward_item_counts'] = []

    def after_step(self, state):
        idx = state.get('idx')
        padding_value = state.get('padding_value', -1)

        if idx is None:
            return

        is_valid = (idx != padding_value)
        items_this_batch = is_valid.sum(dim=0).long()

        self.forward_item_counts += items_this_batch.cpu()

    def after_test(self, state):
        """Logs the current counts to the data dictionary."""
        state['data']['forward_item_counts'].append(self.forward_item_counts.clone().numpy())
        
class BackwardPassCounter(Interceptor):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        self.backward_pass_counts = torch.zeros(state['n_networks'], dtype=torch.long)
        state['data']['backward_pass_counts'] = []

    def after_step(self, state):
        per_sample_loss = state.get('per_sample_losses')
        if per_sample_loss is None: return
        
        was_updated_mask = (per_sample_loss > 1e-9) # eps for rounding errors
        
        updates_this_batch = was_updated_mask.any(dim=0).long()
        self.backward_pass_counts += updates_this_batch.cpu()

    def after_test(self, state):
        state['data']['backward_pass_counts'].append(self.backward_pass_counts.clone().numpy())

class BackwardItemCounter(Interceptor):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        self.backward_item_counts = torch.zeros(state['n_networks'], dtype=torch.long)
        state['data']['backward_item_counts'] = []

    def after_step(self, state):
        per_sample_loss = state.get('per_sample_losses')
        if per_sample_loss is None: return

        was_updated_mask = (per_sample_loss > 1e-9) # eps for rounding errors

        updated_items_this_batch = was_updated_mask.sum(dim=0).long()
        self.backward_item_counts += updated_items_this_batch.cpu()

    def after_test(self, state):
        state['data']['backward_item_counts'].append(self.backward_item_counts.clone().numpy())
        
class PerSampleBackwardCounter(Interceptor):
    def __init__(self, max_samples):
        super().__init__()
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")
        self.max_samples = max_samples

    def before_train(self, state):
        n_networks = state['n_networks']
        self.per_sample_backward_counts = torch.zeros((self.max_samples, n_networks), dtype=torch.long)
        state['data']['per_sample_backward_counts'] = self.per_sample_backward_counts

    def after_step(self, state):
        per_sample_loss = state.get('per_sample_losses').cpu()

        idx = state.get('idx').cpu()

        if per_sample_loss is None or idx is None:
            return

        # get indices of valid samples that contributed to the loss
        update_coords = torch.where(
            (per_sample_loss > 1e-9) & (idx != state.get('padding_value', -1)) # eps for rounding errors
        )
        if update_coords[0].numel() == 0:
            return 

                            #    Gemini-Pro 2.5 tip
                            #             |
                            #             V
        self.per_sample_backward_counts.index_put_(
            (idx[update_coords], update_coords[1]),
            torch.tensor(1, device='cpu'),
            accumulate=True
        )

    def after_train(self, state):
        state['data']['per_sample_backward_counts'] = self.per_sample_backward_counts.clone().numpy()
    
    # interface with replay
    def get_samples(self):
        return self.per_sample_backward_counts
    
    # interface with hard mining sampler     
    def get_samples_as_list_per_network(self, i):
        return np.where(self.per_sample_backward_counts.T[i] > 0)[0].tolist()
    
# Gemini-Pro
class WeightStatsTracker(Interceptor):
    """
    An interceptor that tracks a user-specified list of statistics for model
    'weights' and/or 'gradients' with configurable granularity.
    """
    def __init__(self, tensors_to_track=['weights', 'gradients'], 
                 stats_to_track=['mean', 'std', 'norm'], granularity='global'):
        super().__init__()
        
        # --- Validation ---
        if granularity not in ['global', 'layerwise']:
            raise ValueError(f"granularity must be 'global' or 'layerwise', but got {granularity}")
        supported_tensors = ['weights', 'gradients']
        if not all(t in supported_tensors for t in tensors_to_track):
            raise ValueError(f"tensors_to_track can only contain: {supported_tensors}")
        supported_stats = ['mean', 'std', 'min', 'max', 'norm']
        if not all(stat in supported_stats for stat in stats_to_track):
            raise ValueError(f"stats_to_track can only contain: {supported_stats}")
        
        self.tensors_to_track = tensors_to_track
        self.stats_to_track = stats_to_track
        self.granularity = granularity
        self.last_grads = None

    def before_train(self, state):
        """Initializes the data structures for logging based on granularity."""
        for tensor_type in self.tensors_to_track:
            log_key = f"{tensor_type}_stats_{self.granularity}"
            if self.granularity == 'global':
                state['data'][log_key] = {stat: [] for stat in self.stats_to_track}
            else: # layerwise
                state['data'][log_key] = {stat: {name: [] for name, _ in state['model'].named_parameters()} for stat in self.stats_to_track}

    def before_step(self, state):
        """Captures named gradients before the optimizer step."""
        if 'gradients' in self.tensors_to_track:
            self.last_grads = {n: p.grad.clone() for n, p in state['model'].named_parameters() if p.grad is not None}

    def _calculate_stats(self, named_tensors, state):
        """Helper to compute stats at the specified granularity, per network."""
        if not named_tensors: return {}

        if self.granularity == 'global':
            n_networks = state['n_networks']
            device = next(iter(named_tensors.values())).device
            
            # Reorganize tensors by network
            params_per_network = [[] for _ in range(n_networks)]
            for name, tensor in named_tensors.items():
                split_tensors = torch.unbind(tensor, dim=0)
                for i, t_slice in enumerate(split_tensors):
                    params_per_network[i].append(t_slice.flatten())

            # Calculate stats for each network's full parameter vector
            stats_per_network = {stat: [] for stat in self.stats_to_track}
            for i in range(n_networks):
                if not params_per_network[i]: continue
                vec = torch.cat(params_per_network[i]).detach().abs()
                
                if 'mean' in self.stats_to_track: stats_per_network['mean'].append(vec.mean().item())
                if 'std' in self.stats_to_track: stats_per_network['std'].append(vec.std().item())
                if 'min' in self.stats_to_track: stats_per_network['min'].append(vec.min().item())
                if 'max' in self.stats_to_track: stats_per_network['max'].append(vec.max().item())
                if 'norm' in self.stats_to_track: stats_per_network['norm'].append(torch.linalg.norm(vec).item())

            # Convert lists of stats to tensors
            final_stats = {
                stat: torch.tensor(values, device=device) 
                for stat, values in stats_per_network.items() if values
            }
            return final_stats

        else: # layerwise
            layer_stats = {}
            for name, tensor in named_tensors.items():
                stats = {}
                # Calculate stats over all dimensions except the first (network) dimension
                reduce_dims = tuple(range(1, tensor.ndim))
                
                if reduce_dims:
                    if 'mean' in self.stats_to_track: stats['mean'] = tensor.mean(dim=reduce_dims)
                    if 'std' in self.stats_to_track: stats['std'] = tensor.std(dim=reduce_dims)
                    if 'min' in self.stats_to_track: stats['min'] = tensor.amin(dim=reduce_dims)
                    if 'max' in self.stats_to_track: stats['max'] = tensor.amax(dim=reduce_dims)
                    if 'norm' in self.stats_to_track: stats['norm'] = torch.linalg.norm(tensor, dim=reduce_dims)
                else: # Handle 1D batched tensors (e.g. bias for single output neuron)
                    if 'mean' in self.stats_to_track: stats['mean'] = tensor.clone()
                    if 'std' in self.stats_to_track: stats['std'] = torch.zeros_like(tensor)
                    if 'min' in self.stats_to_track: stats['min'] = tensor.clone()
                    if 'max' in self.stats_to_track: stats['max'] = tensor.clone()
                    if 'norm' in self.stats_to_track: stats['norm'] = tensor.abs()

                layer_stats[name] = stats
            return layer_stats

    def _log_stats(self, log_key, stats_data, state):
        """Helper to log stats based on the granularity."""
        if self.granularity == 'global':
            for stat_name, value_tensor in stats_data.items():
                if stat_name in state['data'][log_key]:
                    state['data'][log_key][stat_name].append(value_tensor.cpu().numpy())
        else: # layerwise
            rearranged_data = {stat: {} for stat in self.stats_to_track}
            for param_name, stats_dict in stats_data.items():
                for stat_name, value_tensor in stats_dict.items():
                    rearranged_data[stat_name][param_name] = value_tensor
            
            for stat_name, param_dict in rearranged_data.items():
                for param_name, value_tensor in param_dict.items():
                    if stat_name in state['data'][log_key] and param_name in state['data'][log_key][stat_name]:
                        state['data'][log_key][stat_name][param_name].append(value_tensor.cpu().numpy())

    @torch.no_grad()
    def after_test(self, state):
        """Calculates and logs stats at the end of an epoch."""
        if 'weights' in self.tensors_to_track:
            named_weights = {n: p.data for n, p in state['model'].named_parameters()}
            weight_stats = self._calculate_stats(named_weights, state)
            self._log_stats(f"weights_stats_{self.granularity}", weight_stats, state)
        
        if 'gradients' in self.tensors_to_track and self.last_grads:
            grad_stats = self._calculate_stats(self.last_grads, state)
            self._log_stats(f"gradients_stats_{self.granularity}", grad_stats, state)
            self.last_grads = None

########################## NETWORK LOGIC MODIFICAITON ##########################
class MaskLinear(Interceptor):
    """
    Transforms a single BatchLinear module into a masked layer using Interceptor hooks:
    1. Generates and registers masks as buffers in __init__ (on CPU).
    2. Uses before_*_forward hooks to apply the mask before computation.
    3. Registers gradient hooks in before_train to enforce zero gradients during backward pass.
    """
    def __init__(self, layer, mask_config):
        super().__init__()
        
        # --- Mandatory Type Check ---
        if not isinstance(layer, BatchLinear):
            raise TypeError(
                f"MaskLinear received an invalid module type: "
                f"Expected 'BatchLinear', got '{type(layer).__name__}'. "
                "This interceptor only supports BatchLinear modules."
            )
        self.layer = layer
        
        self.layer.register_buffer('weight_mask', mask_config['weight_mask'].to(layer.weights.device))
        self.layer.register_buffer('bias_mask', mask_config['bias_mask'].to(layer.weights.device))
        
        self.mask_activities = mask_config['mask_activities']
        self.mask_gradients = mask_config['mask_gradients']
        self._apply_mask()

    @torch.no_grad()
    def _apply_mask(self):
        """
        Applies the mask directly to the parameter data. 
        """
        if self.mask_activities:
            self.layer.weights.data *= self.layer.weight_mask
            self.layer.biases.data *= self.layer.bias_mask
    
    # hooks
    def before_train(self, state):
        """Registers backward hooks now that the model is on its final device."""
        print("Transforming BatchLinear module into a masked layer using Interceptor hooks...")
        
        if hasattr(self, '_grad_hooks_registered') or not self.mask_gradients:
            return
        
        if self.mask_gradients:
            print('grads hooked')
            self.layer.weights.register_hook(lambda grad : grad * self.layer.weight_mask)
            self.layer.biases.register_hook(lambda grad : grad * self.layer.bias_mask)
            
        self._grad_hooks_registered = True
            
    def before_train_forward(self, state):
        """Apply masks before training forward pass."""
        self._apply_mask()

    def before_test_forward(self, state):
        """Apply masks before testing forward pass."""
        self._apply_mask()
        
class AdversarialGradientMasker(Interceptor):
    """
    An interceptor that dynamically masks weights in a BatchLinear layer based on
    anticipated gradients, inspired by the logic of the Competitive optimizer.

    It supports various masking granularities:
    - 'layer_wise_weight': Masks a fraction of connections across the entire layer.
    - 'neuron_wise_weight': Masks a fraction of connections for each neuron individually.
    - 'layer_wise_neuron': Masks entire neurons based on their total gradient contribution.
    """
    def __init__(self, layer, k_fraction, mode='layer_wise_weight', use_largest=True, neuron_dim=None):
        if not hasattr(layer, 'weights'):
            raise ValueError("The provided layer must be a BatchLinear instance.")
        if mode not in ['layer_wise_weight', 'neuron_wise_weight', 'layer_wise_neuron']:
            raise ValueError(f"Mode must be one of ['layer_wise_weight', 'neuron_wise_weight', 'layer_wise_neuron'].")
        if 'neuron' in mode and neuron_dim not in ['incoming', 'outgoing']:
            raise ValueError("neuron_dim must be 'incoming' or 'outgoing' for neuron modes.")
        
        self.layer = layer
        self.k_fraction = self._process_k(k_fraction)
        self.mode = mode
        self.neuron_dim = neuron_dim
        self.use_largest = use_largest
        
        self.original_weights = None
        self.mask = None

    def _process_k(self, k):
        """Validates that k is a valid fraction between 0.0 and 1.0."""
        if isinstance(k, (float, list, np.ndarray)):
            k = torch.tensor(k, dtype=torch.float32)
        if isinstance(k, torch.Tensor):
            if torch.any((k < 0.0) | (k > 1.0)):
                raise ValueError("All values for k_fraction must be between 0.0 and 1.0.")
            return k
        raise TypeError("k_fraction must be a float, list, np.ndarray, or torch.Tensor")

    def before_train_forward(self, state):
        """Hook for performing the preliminary pass, calculating and applying the mask."""
        model = state['model']
        x, y, idx, criterion = state['x'], state['y'], state['idx'], state['criterion']
        
        # 1. Preliminary Forward and Backward Pass
        model.eval()
        with torch.enable_grad():
            model.zero_grad()
            y_pred_naive = model(x)
            per_sample_supervised_loss = criterion(y_pred_naive, y, idx, state['padding_value'])
            mask = (idx != state['padding_value'])
            n_valid_samples = mask.sum(0)
            loss = per_sample_supervised_loss.sum(0) / (n_valid_samples + 1e-12) # average 
            loss.sum().backward()
        model.train()

        grads = self.layer.weights.grad
        if grads is None: return

        # 2. Generate and store the mask
        self.mask = self._generate_mask(grads)

        # 3. Apply the mask
        with torch.no_grad():
            self.original_weights = self.layer.weights.data.clone()
            self.layer.weights.data *= self.mask

    def before_step(self, state):
        """
        Masks the gradients *before* the optimizer steps.
        This ensures that only unmasked weights will receive an update.
        """
        if self.mask is not None and self.layer.weights.grad is not None:
            with torch.no_grad():
                # Zero out gradients for the weights that were masked
                self.layer.weights.grad.data *= self.mask

    def after_step(self, state):
        """
        Restores the *original* values for the masked-out weights,
        while keeping the *newly updated* values for the unmasked weights.
        """
        if self.original_weights is not None and self.mask is not None:
            with torch.no_grad():
                # Get the weights tensor, which now contains the
                # updated values for the unmasked parts
                updated_weights = self.layer.weights.data
                
                # (1.0 - self.mask) creates a mask for the parts we want to restore
                # (self.mask) creates a mask for the parts we want to keep
                self.layer.weights.data = (updated_weights * self.mask) + \
                                          (self.original_weights * (1.0 - self.mask))
            
            # Clear state for the next iteration
            self.original_weights, self.mask = None, None

    def _prepare_k_per_batch(self, p):
        """Expands the k_fraction tensor to a per-network value."""
        batch_size = p.shape[0]
        device = p.device
        k = self.k_fraction.to(device)

        if k.ndim == 0:  # Scalar k
            return k.expand(batch_size)
        elif k.ndim == 1:  # 1D k
            if len(k) != batch_size:
                raise ValueError(f"1D k_fraction length ({len(k)}) must match batch size ({batch_size}).")
            return k
        else:
            raise ValueError("k_fraction must be a scalar or 1D tensor.")

    def _generate_mask(self, grads):
        """
        Generates the adversarial mask based on the specified mode.
        k_fraction now refers to the fraction of weights/neurons to KEEP.
        Ensures exactly k-fraction items are kept by using index-based scattering.
        Randomly breaks ties by adding a small amount of noise.
        """
        n_linears = grads.shape[0]
        k_per_batch = self._prepare_k_per_batch(grads) # k_fraction = fraction to KEEP
        grads_abs = grads.abs()

        # --- NEW: Add small random noise to break ties ---
        # This ensures that if two gradients are identical (e.g., 0.5 and 0.5),
        # the one chosen for the top-k is selected randomly.
        # The noise is scaled to be extremely small and not affect non-tied items.
        noise = torch.rand_like(grads_abs) * 1e-6 
        grads_to_sort = grads_abs + noise
        # --- END NEW ---

        # Determine sort order
        # use_largest=True -> keep largest -> sort descending
        # use_largest=False -> keep smallest -> sort ascending
        descending = self.use_largest

        if self.mode == 'layer_wise_weight':
            grads_flat = grads_to_sort.view(n_linears, -1) # Use grads_to_sort
            num_elements = grads_flat.shape[1]
            num_to_keep = torch.ceil(k_per_batch * num_elements).long()
            num_to_keep.clamp_(max=num_elements)
            
            mask_flat = torch.zeros_like(grads_flat)
            _, sorted_indices = torch.sort(grads_flat, dim=1, descending=descending) # Sort the noisy tensor
            
            arange_mask = torch.arange(num_elements, device=grads.device).expand_as(grads_flat)
            k_mask = arange_mask < num_to_keep.unsqueeze(1)
            
            mask_flat.scatter_(1, sorted_indices, k_mask.float())
            return mask_flat.view_as(grads)

        elif self.mode == 'neuron_wise_weight':
            selection = grads_to_sort.clone() # Use grads_to_sort
            if self.neuron_dim == 'incoming':
                selection = selection.transpose(1, 2)
            
            n_linears, n_neurons, n_synapses = selection.shape[0], selection.shape[1], selection.shape[2]
            selection_flat = selection.reshape(-1, n_synapses) 

            k_expanded = k_per_batch.repeat_interleave(n_neurons) 
            num_to_keep = torch.ceil(k_expanded * n_synapses).long()
            num_to_keep.clamp_(max=n_synapses) 

            _, sorted_indices = torch.sort(selection_flat, dim=1, descending=descending) # Sort the noisy tensor

            arange_mask = torch.arange(n_synapses, device=grads.device).expand_as(selection_flat)
            k_mask = arange_mask < num_to_keep.unsqueeze(1) 

            mask_flat = torch.zeros_like(selection_flat)
            mask_flat.scatter_(1, sorted_indices, k_mask.float())
            
            mask = mask_flat.reshape(n_linears, n_neurons, n_synapses)
            return mask.transpose(1, 2) if self.neuron_dim == 'incoming' else mask

        elif self.mode == 'layer_wise_neuron':
            sum_dim = 2 if self.neuron_dim == 'outgoing' else 1
            # We must sum the *original* grads, but sort using the *noisy* ones
            neuron_demand_abs = grads_abs.sum(dim=sum_dim)
            neuron_demand_noisy = grads_to_sort.sum(dim=sum_dim) # Shape (n_linears, n_neurons)
            
            n_neurons = neuron_demand_abs.shape[1]
            num_neurons_to_keep = torch.ceil(k_per_batch * n_neurons).long()
            num_neurons_to_keep.clamp_(max=n_neurons)
            
            # Sort using the noisy demands to get randomized indices
            _, sorted_indices = torch.sort(neuron_demand_noisy, dim=1, descending=descending)
            
            # Create a boolean mask for the top k indices
            arange_mask = torch.arange(n_neurons, device=grads.device).expand_as(neuron_demand_abs)
            k_mask = arange_mask < num_neurons_to_keep.unsqueeze(1)

            # Create the final neuron mask (all zeros)
            neuron_mask = torch.zeros_like(neuron_demand_abs, dtype=torch.float)
            
            # Scatter 1s into the mask at the positions of the winning neurons
            neuron_mask.scatter_(1, sorted_indices, k_mask.float())

            # Expand the neuron mask to the full weight tensor shape
            mask = neuron_mask.unsqueeze(sum_dim).expand_as(grads)
            # plt.imshow(neuron_mask[-1].detach().cpu().reshape(28,28))
            # plt.show()
            return mask.float()

class DynamicActivityMasker(Interceptor):
    """
    An interceptor that dynamically masks weights in a BatchLinear layer based on
    their magnitude ("activity") at the start of each step.

    This implements a form of dynamic, temporary magnitude pruning.
    The hooks ensure only the "kept" weights are trained for this step.
    
    It supports:
    - 'layer_wise_weight': Keeps k% of connections across the entire layer.
    - 'neuron_wise_weight': Keeps k% of connections for each neuron individually.
    - 'layer_wise_neuron': Keeps k% of entire neurons based on their total weight magnitude.
    """
    def __init__(self, layer, k_fraction, mode='layer_wise_weight', use_largest=True, neuron_dim=None):
        if not hasattr(layer, 'weights'):
            raise ValueError("The provided layer must be a BatchLinear instance.")
        if mode not in ['layer_wise_weight', 'neuron_wise_weight', 'layer_wise_neuron']:
            raise ValueError(f"Mode must be one of ['layer_wise_weight', 'neuron_wise_weight', 'layer_wise_neuron'].")
        if 'neuron' in mode and neuron_dim not in ['incoming', 'outgoing']:
            raise ValueError("neuron_dim must be 'incoming' or 'outgoing' for neuron modes.")
            
        self.layer = layer
        self.k_fraction = self._process_k(k_fraction)
        self.mode = mode
        self.neuron_dim = neuron_dim
        self.use_largest = use_largest
        
        self.original_weights = None
        self.mask = None

    def _process_k(self, k):
        """Validates that k is a valid fraction between 0.0 and 1.0."""
        if isinstance(k, (float, list, np.ndarray)):
            k = torch.tensor(k, dtype=torch.float32)
        if isinstance(k, torch.Tensor):
            if torch.any((k < 0.0) | (k > 1.0)):
                raise ValueError("All values for k_fraction must be between 0.0 and 1.0.")
            return k
        raise TypeError("k_fraction must be a float, list, np.ndarray, or torch.Tensor")

    def before_train_forward(self, state):
        """
        Hook for calculating the activity mask and applying it.
        This is now much simpler as it doesn't need a preliminary pass.
        """
        # 1. Get the selection tensor (the weights themselves)
        selection_tensor = self.layer.weights.data
        if selection_tensor is None: return

        # 2. Generate and store the mask based on weight magnitudes
        self.mask = self._generate_mask(selection_tensor)

        # 3. Apply the mask
        with torch.no_grad():
            self.original_weights = self.layer.weights.data.clone()
            self.layer.weights.data *= self.mask

    def before_step(self, state):
        """
        Masks the gradients *before* the optimizer steps.
        This ensures that only unmasked weights will receive an update.
        """
        if self.mask is not None and self.layer.weights.grad is not None:
            with torch.no_grad():
                # Zero out gradients for the weights that were masked
                self.layer.weights.grad.data *= self.mask

    def after_step(self, state):
        """
        Restores the *original* values for the masked-out weights,
        while keeping the *newly updated* values for the unmasked weights.
        """
        if self.original_weights is not None and self.mask is not None:
            with torch.no_grad():
                updated_weights = self.layer.weights.data
                
                # Restore original weights for masked-out parts
                self.layer.weights.data = (updated_weights * self.mask) + \
                                          (self.original_weights * (1.0 - self.mask))
            
            # Clear state for the next iteration
            self.original_weights, self.mask = None, None

    def _prepare_k_per_batch(self, p):
        """Expands the k_fraction tensor to a per-network value."""
        batch_size = p.shape[0]
        device = p.device
        k = self.k_fraction.to(device)

        if k.ndim == 0:  # Scalar k
            return k.expand(batch_size)
        elif k.ndim == 1:  # 1D k
            if len(k) != batch_size:
                raise ValueError(f"1D k_fraction length ({len(k)}) must match batch size ({batch_size}).")
            return k
        else:
            raise ValueError("k_fraction must be a scalar or 1D tensor.")

    def _generate_mask(self, selection_tensor):
        """
        Generates the mask based on the selection tensor (weight magnitudes).
        k_fraction now refers to the fraction of weights/neurons to KEEP.
        """
        n_linears = selection_tensor.shape[0]
        k_per_batch = self._prepare_k_per_batch(selection_tensor) # k_fraction = fraction to KEEP
        selection_abs = selection_tensor.abs()

        # Add small random noise to break ties
        noise = torch.rand_like(selection_abs) * 1e-6 
        tensor_to_sort = selection_abs + noise

        # Determine sort order
        descending = self.use_largest

        if self.mode == 'layer_wise_weight':
            flat_tensor = tensor_to_sort.view(n_linears, -1)
            num_elements = flat_tensor.shape[1]
            num_to_keep = torch.ceil(k_per_batch * num_elements).long()
            num_to_keep.clamp_(max=num_elements)
            
            mask_flat = torch.zeros_like(flat_tensor)
            _, sorted_indices = torch.sort(flat_tensor, dim=1, descending=descending)
            
            arange_mask = torch.arange(num_elements, device=selection_tensor.device).expand_as(flat_tensor)
            k_mask = arange_mask < num_to_keep.unsqueeze(1)
            
            mask_flat.scatter_(1, sorted_indices, k_mask.float())
            return mask_flat.view_as(selection_tensor)

        elif self.mode == 'neuron_wise_weight':
            selection = tensor_to_sort.clone()
            if self.neuron_dim == 'incoming':
                selection = selection.transpose(1, 2)
            
            n_linears, n_neurons, n_synapses = selection.shape[0], selection.shape[1], selection.shape[2]
            selection_flat = selection.reshape(-1, n_synapses) 

            k_expanded = k_per_batch.repeat_interleave(n_neurons) 
            num_to_keep = torch.ceil(k_expanded * n_synapses).long()
            num_to_keep.clamp_(max=n_synapses) 

            _, sorted_indices = torch.sort(selection_flat, dim=1, descending=descending)

            arange_mask = torch.arange(n_synapses, device=selection_tensor.device).expand_as(selection_flat)
            k_mask = arange_mask < num_to_keep.unsqueeze(1) 

            mask_flat = torch.zeros_like(selection_flat)
            mask_flat.scatter_(1, sorted_indices, k_mask.float())
            
            mask = mask_flat.reshape(n_linears, n_neurons, n_synapses)
            return mask.transpose(1, 2) if self.neuron_dim == 'incoming' else mask

        elif self.mode == 'layer_wise_neuron':
            sum_dim = 2 if self.neuron_dim == 'outgoing' else 1
            # Sum original magnitudes, sort using noisy ones
            neuron_demand_abs = selection_abs.sum(dim=sum_dim)
            neuron_demand_noisy = tensor_to_sort.sum(dim=sum_dim)
            
            n_neurons = neuron_demand_abs.shape[1]
            num_neurons_to_keep = torch.ceil(k_per_batch * n_neurons).long()
            num_neurons_to_keep.clamp_(max=n_neurons)
            
            _, sorted_indices = torch.sort(neuron_demand_noisy, dim=1, descending=descending)
            
            arange_mask = torch.arange(n_neurons, device=selection_tensor.device).expand_as(neuron_demand_abs)
            k_mask = arange_mask < num_neurons_to_keep.unsqueeze(1)

            neuron_mask = torch.zeros_like(neuron_demand_abs, dtype=torch.float)
            neuron_mask.scatter_(1, sorted_indices, k_mask.float())

            mask = neuron_mask.unsqueeze(sum_dim).expand_as(selection_tensor)
            return mask.float()

############################# AUXILLARY LOSSES ################################
"""
These Interceptors make modifications to the loss function before computing 
gradints and are therefore useful for auxillary losses like L1/L2 regularization.
"""
class L1Regularizer(Interceptor):
    def __init__(self, lambda_l1=0.01, reference_provider=None):
        super().__init__()
        if isinstance(lambda_l1, (list, np.ndarray)):
            self.lambda_l1 = torch.as_tensor(lambda_l1, dtype=torch.float32)
        else:
            self.lambda_l1 = lambda_l1
        
        if torch.is_tensor(self.lambda_l1) and (self.lambda_l1 < 0).any():
             raise ValueError("All lambda_l1 values must be non-negative.")
        elif isinstance(self.lambda_l1, (float, int)) and self.lambda_l1 < 0:
            raise ValueError("lambda_l1 must be a non-negative number.")
            
        self.reference_provider = reference_provider

    def after_train_forward(self, state):
        n_networks = state['n_networks']
        device = state['device']

        if torch.is_tensor(self.lambda_l1):
            lambda_tensor = self.lambda_l1.to(device)
            if lambda_tensor.ndim == 1 and len(lambda_tensor) != n_networks:
                raise ValueError(f"Length of lambda_l1 tensor ({len(lambda_tensor)}) must match n_networks ({n_networks}).")
        elif isinstance(self.lambda_l1, (float, int)):
            if self.lambda_l1 == 0:
                return
            lambda_tensor = self.lambda_l1
        else:
            return


        reference_point = None
        if self.reference_provider is not None:
            reference_point = self.reference_provider.get_parameters()

        l1_penalty = torch.zeros(n_networks, device=device)
        
        for name, param in state['model'].named_parameters():
            if param.requires_grad:
                if reference_point is not None:
                    if name not in reference_point:
                        continue
                    ref_param = reference_point[name].to(device)
                    delta = param - ref_param
                else:
                    delta = param # else regularize towards zero
                per_network_norm = delta.abs().sum(dim=tuple(range(1, param.ndim)))
                l1_penalty += per_network_norm
                
        scaled_penalty = l1_penalty * lambda_tensor
        state['loss'] = state['loss'] + scaled_penalty

class L2Regularizer(Interceptor):
    def __init__(self, lambda_l2=0.01, reference_provider=None):
        super().__init__()
        if isinstance(lambda_l2, (list, np.ndarray)):
            self.lambda_l2 = torch.as_tensor(lambda_l2, dtype=torch.float32)
        else:
            self.lambda_l2 = lambda_l2
        

        if torch.is_tensor(self.lambda_l2) and (self.lambda_l2 < 0).any():
             raise ValueError("All lambda_l2 values must be non-negative.")
        elif isinstance(self.lambda_l2, (float, int)) and self.lambda_l2 < 0:
            raise ValueError("lambda_l2 must be a non-negative number.")

        self.reference_provider = reference_provider # usually a param provider

    def after_train_forward(self, state):

        n_networks = state['n_networks']
        device = state['device']

        if torch.is_tensor(self.lambda_l2):
            lambda_tensor = self.lambda_l2.to(device)
            if lambda_tensor.ndim == 1 and len(lambda_tensor) != n_networks:
                raise ValueError(f"Length of lambda_l2 tensor ({len(lambda_tensor)}) must match n_networks ({n_networks}).")
        elif isinstance(self.lambda_l2, (float, int)):
            if self.lambda_l2 == 0:
                return
            lambda_tensor = self.lambda_l2
        else:
            return


        reference_point = None
        if self.reference_provider is not None:
            reference_point = self.reference_provider.get_parameters()

        l2_penalty_sq = torch.zeros(n_networks, device=device)
        
        for name, param in state['model'].named_parameters():
            if param.requires_grad:
                if reference_point is not None:
                    if name not in reference_point:
                        continue
                    ref_param = reference_point[name].to(device)
                    delta = param - ref_param # regularize towards ref.
                else:
                    delta = param # else regularize towards zero

                per_network_sum_sq = (delta ** 2).sum(dim=tuple(range(1, param.ndim)))
                l2_penalty_sq += per_network_sum_sq
        
        scaled_penalty = l2_penalty_sq * lambda_tensor
        
        state['loss'] = state['loss'] + scaled_penalty
                
class Automasking(Interceptor):
    def __init__(self, threshold=0.1, p=1, model=None):
        super().__init__()
        self.model = model
        self.p = p # metric used to deteremine "error"
        self.threshold = threshold
        
    def before_train(self, state):
        if self.model is None:
            self.model = state['model']
    
    @torch.no_grad()
    def before_train_forward(self, state):
        x_prime = self.model(state['x'])
        mask = (x_prime.abs() - state['y'])**self.p > self.threshold
        x_prime[mask] = 0
        state['x'] = x_prime
         
############################# PROVIDERS ################################
"""
Providers "provide" some kind of value that is often used by multiple other
Interceptors. This avoid repeated memory allocation. For instance, if 
multiple Interceptors require the previous parameters, with these they would 
each store previous parameters. Instead we pass a provider to the Interceptor
where they can share their use.
"""
        
 # stops repeated memory (i.e. if multiple observers need previous parameters)
class PreviousParameterProvider(Interceptor):
    def __init__(self, model=None):
        super().__init__() # need model for intialization
        if model is None:
            self.previous_parameters = {} # lazy init
        
    def before_step(self, state): #                          now we need the state
        self.previous_parameters = {n: p.clone().detach() for n, p in state['model'].named_parameters()}
    
    def after_update(self, state):
        self.previous_parameters = {} # save GPU memory
        
    def get_parameters(self):
        return self.previous_parameters
    
class PreviousPreviousParameterProvider(Interceptor):
    def __init__(self, previous_parameter_provider):
        super().__init__()
        self.previous_previous_parameters = {}
        self.provider = previous_parameter_provider
    
    def after_update(self, state):
        self.previous_previous_parameters = {n: p.clone().detach() 
                                    for n, p in self.provider.get_parameters().items()}
        
    def get_parameters(self):
        return self.previous_previous_parameters
        
        # stops repeated memory
class InitialParameterProvider(Interceptor):
    def __init__(self, model=None):
        super().__init__() # need model for intialization
        if model is None:
            self.initial_parameters = {} # lazy init
        else:
            self.initial_parameters = {n: p.clone().detach() for n, p in model.named_parameters()}
        
    def before_train(self, state): #                          now we need the state
        if not self.initial_parameters:
            self.initial_parameters = {n: p.clone().detach() for n, p in state['model'].named_parameters()}
    
    def get_parameters(self):
        return self.initial_parameters
    
############################# ENERGY FUNCS ################################
"""
This Interceptor generalizes the compute energy calculations (Li & van Rossum 2020) and 
usually requires a provider. They have been split into network, layerwise and neuronwise
to get different granularities of energy metrics.
"""    
# kinda wild that you can feed all the energy interceptors as example and then Gemini-Pro 
# can whip this up with surprisingly few corrections needed
class EnergyMetricTracker(Interceptor):
    """
    A generalized interceptor to track the delta ("energy") of network parameters.

    It is highly configurable, allowing for tracking based on:
    - p (int or float): The order of the Lp norm (e.g., 0, 1, 2, 0.5).
    - mode (str): 'displacement' (vs. initial state) or 'cumulative' (step-by-step).
    - granularity (str): 'network', 'layerwise', or 'neuronwise'.
    - components (list): Which parts to track: ['total', 'weight', 'bias'].
    """
    def __init__(self, p, param_provider, mode, granularity='network', components=['total'], energy_direction='incoming'):
        super().__init__()
        # --- Configuration Validation ---
        if not isinstance(p, (int, float)) or p < 0:
            raise ValueError(f"p must be a non-negative number, but got {p}.")
        if mode not in ['energy', 'minimum_energy']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'energy' or 'minimum_energy'.")
        if granularity not in ['network', 'layerwise', 'neuronwise']:
            raise ValueError(f"Invalid granularity: {granularity}. Must be 'network', 'layerwise', or 'neuronwise'.")
        
        self.p = p
        self.provider = param_provider
        self.mode = mode
        self.granularity = granularity
        self.components = [components] if isinstance(components, str) else list(components) # Ensure it's a mutable list
        self.energy_direction = [energy_direction] if isinstance(energy_direction, str) else energy_direction
        
        if not all(c in ['total', 'weight', 'bias'] for c in self.components):
            raise ValueError("components must be a list with options from: 'total', 'weight', 'bias'")

        # --- Handle 'total' ambiguity for neuronwise granularity ---
        if self.granularity == 'neuronwise' and 'total' in self.components:
            print(f"INFO (Tracker p={self.p}, {self.mode}): For 'neuronwise' granularity, 'total' is ambiguous. Defaulting to tracking 'weight' and 'bias' components separately.")
            # Add weight and bias if they aren't there, then remove total.
            if 'weight' not in self.components:
                self.components.append('weight')
            if 'bias' not in self.components:
                self.components.append('bias')
            self.components.remove('total')
            # Remove duplicates if user provided something like ['total', 'weight']
            self.components = sorted(list(set(self.components)))

        self.metric = {}
        self.log_key = self._generate_log_key()

    def _generate_log_key(self):
        mode_prefix = 'minimum_energies' if self.mode == 'minimum_energy' else 'energies'
        return f"{mode_prefix}_l{self.p}_{self.granularity}"

    def _get_layer_names(self, model):
        """Helper to get unique layer prefixes from parameter names."""
        return sorted(list(set([n.split('.')[0] for n, _ in model.named_parameters()])))

    @torch.no_grad()
    def before_train(self, state):
        """Initializes storage for metrics and logs with component substructure."""
        state['data'][self.log_key] = {comp: {} for comp in self.components}

        if self.granularity == 'layerwise':
            for comp in self.components:
                if comp == 'total':
                    param_keys = self._get_layer_names(state['model'])
                    state['data'][self.log_key][comp] = {n: [] for n in param_keys}
                else:
                    param_keys = [n for n, _ in state['model'].named_parameters() if self._is_component(n, comp)]
                    state['data'][self.log_key][comp] = {n: [] for n in param_keys}
        elif self.granularity == 'neuronwise':
            for comp in self.components:
                param_dict = {}
                for n, p in state['model'].named_parameters():
                    if self._is_component(n, comp):
                        if self._is_component(n, 'bias'):
                            param_dict[n] = [] # Biases don't have energy direction
                        else:
                            param_dict[n] = {e_dir: [] for e_dir in self.energy_direction}
                state['data'][self.log_key][comp] = param_dict
        else: # network
             for comp in self.components:
                state['data'][self.log_key][comp] = []
                
        if self.mode == 'energy':
            self._initialize_metric_accumulator(state)
            
    def _initialize_metric_accumulator(self, state):
        """Sets up `self.metric` as a dictionary of component accumulators on the correct device."""
        device = state['device']
        self.metric = {comp: {} for comp in self.components}
        for comp in self.components:
            dtype = torch.long if self.p == 0 else torch.float
            if self.granularity == 'network':
                self.metric[comp] = torch.zeros(state['n_networks'], dtype=dtype, device=device)
            elif self.granularity == 'layerwise':
                self.metric[comp] = {}
                if comp == 'total':
                    for name in self._get_layer_names(state['model']):
                        self.metric[comp][name] = torch.zeros(state['n_networks'], dtype=dtype, device=device)
                else:
                    for n, _ in state['model'].named_parameters():
                        if self._is_component(n, comp):
                            self.metric[comp][n] = torch.zeros(state['n_networks'], dtype=dtype, device=device)
            elif self.granularity == 'neuronwise':
                self.metric[comp] = {}
                for n, p in state['model'].named_parameters():
                    if self._is_component(n, comp):
                        if self._is_component(n, 'bias'):
                            self.metric[comp][n] = torch.zeros(p.shape, dtype=dtype, device=device)
                        else: # Weight
                            self.metric[comp][n] = {}
                            for e_dir in self.energy_direction:
                                shape = None
                                if p.ndim >= 3:
                                    if e_dir == 'incoming': shape = p.shape[:2] # (batch, out_neurons)
                                    elif e_dir == 'outgoing': shape = (p.shape[0],) + p.shape[2:] # (batch, in_neurons, ...)
                                if shape is not None:
                                    self.metric[comp][n][e_dir] = torch.zeros(shape, dtype=dtype, device=device)

    @torch.no_grad()
    def after_step(self, state):
        """For cumulative mode, calculate step-wise delta and accumulate."""
        if self.mode != 'energy':
            return
        
        ref_params = self.provider.get_parameters()
        deltas = self._calculate_deltas(state['model'], ref_params)
        self._accumulate_deltas(deltas)
        
    @torch.no_grad()
    def after_test(self, state):
        """Logs the final computed metric for the epoch/test run."""
        if self.mode == 'energy':
            metric_to_log = self.metric
        else:  # displacement mode calculates the metric from scratch here
            ref_params = self.provider.get_parameters()
            metric_to_log = self._calculate_deltas(state['model'], ref_params)

        self._log_metric(state, metric_to_log)
    
    @torch.no_grad()
    def _calculate_deltas(self, model, ref_params):
        """Core method to compute deltas for weights and biases separately."""
        # Neuronwise requires a different calculation path, handle it first.
        if self.granularity == 'neuronwise':
            results = {comp: {} for comp in self.components}
            for n, p in model.named_parameters():
                delta = p - ref_params[n].to(p.device)
                
                is_bias = self._is_component(n, 'bias')
                
                # Process biases
                if is_bias and 'bias' in self.components:
                    energy = (delta.abs()**self.p)**(1/self.p) if self.p != 0 else (delta != 0).float()
                    results['bias'][n] = energy

                # Process weights
                elif not is_bias and 'weight' in self.components and p.ndim >= 3:
                    neuron_metrics = {}
                    for e_dir in self.energy_direction:
                        sum_dims = tuple(range(2, p.ndim)) if e_dir == 'incoming' else 1
                        sum_of_pows = (delta.abs()**self.p).sum(dim=sum_dims)
                        energy = sum_of_pows**(1/self.p) if self.p != 0 else torch.count_nonzero(delta, dim=sum_dims).float()
                        neuron_metrics[e_dir] = energy
                    results['weight'][n] = neuron_metrics
            return results

        # --- Logic for Network and Layerwise ---
        raw_deltas = {'weight': {}, 'bias': {}}
        for n, p in model.named_parameters():
            comp_type = 'bias' if ('bias' in n) else 'weight'
            ref_p = ref_params[n].to(p.device)
            delta = p - ref_p
            sum_dims = tuple(range(1, p.ndim))
            if self.p == 0:
                raw_deltas[comp_type][n] = torch.count_nonzero(delta, dim=sum_dims)
            else:
                raw_deltas[comp_type][n] = (delta.abs()**self.p).sum(dim=sum_dims)
        
        results = {}
        for comp in self.components:
            if self.granularity == 'network':
                all_param_deltas = list(raw_deltas['weight'].values()) + list(raw_deltas['bias'].values()) if comp == 'total' else list(raw_deltas[comp].values())
                sum_of_pows = torch.stack(all_param_deltas).sum(dim=0) if all_param_deltas else torch.tensor(0.0, device=p.device)
                results[comp] = sum_of_pows**(1 / self.p) if self.p != 0 else sum_of_pows
            
            elif self.granularity == 'layerwise':
                results[comp] = {}
                if comp == 'total':
                    layer_groups = {}
                    all_raw = {**raw_deltas['weight'], **raw_deltas['bias']}
                    for n, raw_delta in all_raw.items():
                        layer_name = n.split('.')[0]
                        if layer_name not in layer_groups: layer_groups[layer_name] = []
                        layer_groups[layer_name].append(raw_delta)
                    for name, raw_list in layer_groups.items():
                        total_raw = torch.stack(raw_list).sum(dim=0)
                        results[comp][name] = total_raw**(1/self.p) if self.p != 0 else total_raw
                else:
                    for n, raw_d in raw_deltas[comp].items():
                        results[comp][n] = raw_d**(1/self.p) if self.p != 0 else raw_d
        return results

    @torch.no_grad()
    def _accumulate_deltas(self, deltas):
        """Accumulates deltas for each component."""
        for comp, delta_val in deltas.items():
            if self.granularity == 'network':
                self.metric[comp] += delta_val
            elif self.granularity == 'layerwise':
                for name, val in delta_val.items():
                    if name in self.metric[comp]: self.metric[comp][name] += val
            elif self.granularity == 'neuronwise':
                for n, val in delta_val.items():
                    if n in self.metric[comp]:
                        if isinstance(val, dict): # Weight
                            for e_dir, e_val in val.items():
                                if e_dir in self.metric[comp][n]:
                                    self.metric[comp][n][e_dir] += e_val
                        else: # Bias
                            self.metric[comp][n] += val

    @torch.no_grad()
    def _log_metric(self, state, metric_to_log):
        """Appends metrics for each component to the state['data'] dictionary."""
        for comp, metric in metric_to_log.items():
            if self.granularity == 'network':
                state['data'][self.log_key][comp].append(metric.clone().cpu().numpy())
            elif self.granularity == 'layerwise':
                for n, val in metric.items():
                    if n in state['data'][self.log_key][comp]:
                        state['data'][self.log_key][comp][n].append(val.clone().cpu().numpy())
            elif self.granularity == 'neuronwise':
                for n, val in metric.items():
                    if n in state['data'][self.log_key][comp]:
                        if isinstance(val, dict): # Weight
                            for e_dir, e_val in val.items():
                                 if e_dir in state['data'][self.log_key][comp][n]:
                                    state['data'][self.log_key][comp][n][e_dir].append(e_val.clone().cpu().numpy())
                        else: # Bias
                            state['data'][self.log_key][comp][n].append(val.clone().cpu().numpy())

    def _is_component(self, name, component_type):
        """Helper to check if a parameter name matches the component type."""
        is_bias = 'bias' in name
        if component_type == 'total':
            return True
        elif component_type == 'weight':
            return not is_bias
        elif component_type == 'bias':
            return is_bias
        return False
                
############################### INTERCEPTOR GRAVEYARD #############################     
"""
Interceptors here have been superseded by other Interceptor above. For instance,
TestLoop now implements the testing accuracy and loss trackers internally since
it keeps track of valid samples and can appropriately normalise values.
"""      
           
class _TestLoopOld(Interceptor):
    """
    An observer that performs a test/validation loop over a given dataset.
    
    This observer listens for the 'on_test_run' event fired by the Trainer.
    """
    
    def __init__(self, name, test_dataloader, criterions):
        self.name = name
        self.test_dataloader = test_dataloader
        self.criterions = {name: crit for name, crit in criterions.items()}
        self.trainer = None # this will be set by the Trainer

    def on_test_run(self, state):
        self.device = state['device']
        # set the current context so other observers know which dataset is being tested.
        state['current_test_context'] = self.name
        
        state['model'].eval()
        with torch.no_grad():
            for (x, y, idx) in self.test_dataloader:
                batch_size = x.size(0)
                x, y, idx = x.to(self.device), y.to(self.device), idx.to(self.device)
                if len(x.shape) < 3:
                    x = x.unsqueeze(1)
                    y = y.unsqueeze(1).repeat((1, state['n_networks'], 1))
                    idx = idx.unsqueeze(1).repeat(1, state['n_networks'])

                if self.trainer:
                    self.trainer._fire_event('before_test_forward')
                
                y_hat = state['model'](x)


                batch_losses = {name: crit(y_hat, y, idx, state['padding_value'])
                                    for name, crit in self.criterions.items()
                                }
                batch_accuracy = (y_hat.argmax(-1) == y.argmax(-1)).sum(0)
                
                # use the unique name to store results in the state
                
                state[f'{self.name}_losses'] = batch_losses
                state[f'{self.name}_accuracies'] = batch_accuracy

                # fire event after forward pass
                if self.trainer:
                    self.trainer._fire_event('after_test_forward')

        state['model'].train()     
                
class RunningLossTracker(Interceptor):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        self.train_loss_accumulator = torch.zeros((state['n_networks'],))
        state['data']['running_losses'] = []

    def after_update(self, state):
        self.train_loss_accumulator += state['running_loss'].clone().detach().cpu()

    def after_test(self, state):
        state['data']['running_losses'].append(self.train_loss_accumulator.clone().numpy())
        self.train_loss_accumulator = torch.zeros((state['n_networks'],))

# TODO: make testing tracking only store arrays for criterions associated with particular test loops
class TestingLossTracker(Interceptor):
    def __init__(self, test_criterion_map):
        super().__init__()
        if not isinstance(test_criterion_map, dict):
            raise TypeError("test_criterion_map must be a dictionary.")
            
        self.test_criterion_map = test_criterion_map
        self.test_names = list(self.test_criterion_map.keys())
        for test in self.test_names:
            if type(self.test_criterion_map[test]) is str:
                self.test_criterion_map[test] = [self.test_criterion_map[test]]

    def before_train(self, state):
        self.test_losses = {
            test_name: {crit: torch.zeros((state['n_networks'],)) for crit in crit_list}
            for test_name, crit_list in self.test_criterion_map.items()
        }
        
        state['data']['test_losses'] = {
            test_name: {crit: [] for crit in crit_list}
            for test_name, crit_list in self.test_criterion_map.items()
        }

    def after_test_forward(self, state):
        current_test_name = state['current_test_context']
        if current_test_name in self.test_names:
            losses = state[f'{current_test_name}_losses']
            for criterion_name in losses:
                if criterion_name in self.test_criterion_map[current_test_name]:
                    loss_value = losses[criterion_name]
                    self.test_losses[current_test_name][criterion_name] += loss_value.clone().detach().cpu()

    def after_test(self, state):
        for test_name in self.test_losses:
            for criterion_name in self.test_losses[test_name]:
                accumulated_loss = self.test_losses[test_name][criterion_name]
                state['data']['test_losses'][test_name][criterion_name].append(accumulated_loss.clone().numpy())
            self._reset_test(test_name, state)
            
    def _reset_test(self, test_name, state):
        specific_criteria = self.test_criterion_map[test_name]
        self.test_losses[test_name] = {
            crit: torch.zeros((state['n_networks'],)) for crit in specific_criteria
        }

class RunningAccuracyTracker(Interceptor):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        self.train_accuracy_accumulator = torch.zeros((state['n_networks'],))
        state['data']['running_accuracies'] = []

    def after_update(self, state):
        self.train_accuracy_accumulator += state['running_accuracy'].clone().detach().cpu()

    def after_test(self, state):
        state['data']['running_accuracies'].append(self.train_accuracy_accumulator.clone().numpy())
        self.train_accuracy_accumulator = torch.zeros((state['n_networks'],))

class TestingAccuracyTracker(Interceptor):
    def __init__(self, test_names):
        super().__init__()
        if type(test_names) is not list:
            test_names = [test_names]
        self.test_names = test_names

    def before_train(self, state):
        self.test_accuracies = {
            name: torch.zeros((state['n_networks'],)) for name in self.test_names
        }
        
        state['data']['test_accuracies'] = {
            name: [] for name in self.test_names
        }

    def after_test_forward(self, state):
        current_test_name = state['current_test_context']
        if current_test_name in self.test_names:
            accuracy= state[f'{current_test_name}_accuracies']
            self.test_accuracies[current_test_name] += accuracy.clone().detach().cpu()

    def after_test(self, state):
        for test_name in self.test_accuracies:
            accuracy = self.test_accuracies[test_name]
            state['data']['test_accuracies'][test_name].append(accuracy.clone().numpy())
            self._reset_test(test_name, state)
            
    def _reset_test(self, test_name, state):
        self.test_accuracies[test_name] = torch.zeros((state['n_networks'],))
        
class LossTracker(Interceptor):
    def before_train(self, state):
        self.train_loss = torch.zeros((state['n_networks'],))
        self.test_loss = torch.zeros((state['n_networks'],))
        state['data']['running_losses'] = []
        state['data']['test_losses'] = []
        
    def _reset_train(self, state):
        self.train_loss = torch.zeros((state['n_networks'],))
        
    def _reset_test(self, state):
        self.test_loss = torch.zeros((state['n_networks'],))
        
    def after_update(self, state):
        self.train_loss += state['running_loss'].clone().detach().cpu()
        
    def after_test_forward(self, state):
        self.test_loss += state['test_loss'].clone().detach().cpu()
        
    def after_test(self, state):
        state['data']['test_losses'].append(self.test_loss.clone().detach().cpu().numpy())
        self._reset_test(state)
        state['data']['running_losses'].append(self.train_loss.clone().detach().cpu().numpy())
        self._reset_train(state)
        
class AccuracyTracker(Interceptor):
    def before_train(self, state):
        self.train_accuracy = torch.zeros((state['n_networks'],))
        self.test_accuracy = torch.zeros((state['n_networks'],))
        state['data']['running_accuracies'] = []
        state['data']['test_accuracies'] = []
        
    def _reset_train(self, state):
        self.train_accuracy = torch.zeros((state['n_networks'],))
        
    def _reset_test(self, state):
        self.test_accuracy = torch.zeros((state['n_networks'],))
        
    def after_update(self, state):
        self.train_accuracy += state['running_accuracy'].clone().detach().cpu()
        
    def after_test_forward(self, state):
        self.test_accuracy += state['test_accuracy'].clone().detach().cpu()
        
    def after_test(self, state):
        state['data']['test_accuracies'].append(self.test_accuracy.clone().detach().cpu().numpy())
        self._reset_test(state)
        state['data']['running_accuracies'].append(self.train_accuracy.clone().detach().cpu().numpy())
        self._reset_train(state)
        
# TODO: make batch optimizers [Done!]
# This is now performed by batch_optimizers
#              |
class PerNetworkLearningRate(Interceptor):
    def __init__(self, lr_scales):
        super().__init__()
        if not isinstance(lr_scales, torch.Tensor):
            lr_scales = torch.tensor(lr_scales, dtype=torch.float32)
        self.lr_scales = lr_scales

    def before_train(self, state):

        n_networks = state['n_networks']
        if len(self.lr_scales) != n_networks:
            raise ValueError(
                f"The number of lr_scales ({len(self.lr_scales)}) must match "
                f"the number of networks ({n_networks})."
            )
            
        print("INFO: Registering per-network learning rate hooks.")
        for p in state['model'].parameters():
            if p.requires_grad:
                p.register_hook(self._create_hook())

    def _create_hook(self):
        def hook(grad):
            # reshape lr_scales to broadcast correctly with the gradient tensor
            # i.e. from [N] to [N, 1, 1] for a grad of [N, C_out, C_in]
            dims_to_add = grad.ndim - 1
            lr_scales_reshaped = self.lr_scales.view(-1, *([1] * dims_to_add))
            
            return grad * lr_scales_reshaped.to(grad.device)
            
        return hook
    
class MinimumEnergyL1NetworkTracker(Interceptor):
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train(self, state):
        self.minimum_energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l1'] = []
        
    def after_test(self, state):
        delta = torch.zeros((state['n_networks'])).to(state['device'])
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n] 
            delta += (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))
        self.minimum_energy_l1 = delta.detach().cpu()
        state['data']['minimum_energies_l1'].append(self.minimum_energy_l1.clone().numpy())
 
class EnergyL1NetworkTracker(Interceptor):
    def __init__(self, previous_parameter_provider):
        self.provider = previous_parameter_provider
    
    def before_train(self, state):
        self.energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['energies_l1'] = []
        
    def after_step(self, state):       # sum all but the network batch dim (i.e. handles weights or biases)
                                   #                        |
        for n, p in state['model'].named_parameters(): #    |
            prev_p = self.provider.previous_parameters[n] # V
            delta = (p - prev_p).abs().sum(dim=tuple(range(1, p.ndim)))
            self.energy_l1 += delta.detach().cpu()
        
    def after_test(self, state):
        state['data']['energies_l1'].append(self.energy_l1.clone().numpy())
        
class MinimumEnergyL1LayerwiseTracker(Interceptor):
    def __init__(self, initial_parameter_provider):
        super().__init__()
        self.provider = initial_parameter_provider
    
    def before_train(self, state):
        state['data']['minimum_energies_l1_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
        
    def after_test(self, state):
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n].to(p.device)
            delta = (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))
            state['data']['minimum_energies_l1_layerwise'][n].append(delta.detach().cpu().clone().numpy())

class EnergyL1LayerwiseTracker(Interceptor):
    def __init__(self, previous_parameter_provider):
        super().__init__()
        self.provider = previous_parameter_provider

    def before_train(self, state):
        self.energy_l1_layerwise = {
            n: torch.zeros((state['n_networks'],)) for n, _ in state['model'].named_parameters()
        }
        state['data']['energies_l1_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
        
    def after_step(self, state):
        for n, p in state['model'].named_parameters():
            prev_p = self.provider.previous_parameters[n].to(p.device)
            delta = (p - prev_p).abs().sum(dim=tuple(range(1, p.ndim)))

            self.energy_l1_layerwise[n] += delta.detach().cpu()
            
    def after_test(self, state):
        for n, energy in self.energy_l1_layerwise.items():
            state['data']['energies_l1_layerwise'][n].append(energy.clone().numpy())
            
class MinimumEnergyL1NeuronwiseTracker(Interceptor):
    def __init__(self, initial_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = initial_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction
        
    def before_train(self, state):
        state['data']['minimum_energies_l1_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            state['data']['minimum_energies_l1_neuronwise'][n] = {
                e_dir: [] for e_dir in self.energy_direction
            }
    
    def after_test(self, state):
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n].to(p.device)
            delta = (p - init_p).abs()
            
            for e_dir in self.energy_direction:
                energy = None
                if p.ndim <= 2: # for biases or simple 2D params, energy is just the delta
                    energy = delta
                elif p.ndim >= 3: # otherwise for weights (n, j, i, ...)
                    if e_dir == 'incoming':
                        # sum over all input dimensions
                        energy = delta.sum(dim=tuple(range(2, p.ndim)))
                    elif e_dir == 'outgoing':
                        # sum over output dimension (dim 1)
                        energy = delta.sum(dim=1)
                if energy is not None:
                    state['data']['minimum_energies_l1_neuronwise'][n][e_dir].append(energy.detach().cpu().clone().numpy())

class EnergyL1NeuronwiseTracker(Interceptor):

    def __init__(self, previous_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = previous_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction

    def before_train(self, state):
        self.energy_l1_neuronwise = {}
        state['data']['energies_l1_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            self.energy_l1_neuronwise[n] = {}
            state['data']['energies_l1_neuronwise'][n] = {e_dir: [] for e_dir in self.energy_direction}

            for e_dir in self.energy_direction:
                shape = None
                if p.ndim <= 2:
                    shape = p.shape
                elif p.ndim >= 3:
                    if e_dir == 'incoming':
                        shape = p.shape[:2] # (n, j) -> one value per output neuron
                    elif e_dir == 'outgoing':
                        # (n, i, ...) -> one value per input neuron
                        shape = (p.shape[0],) + p.shape[2:]
                
                if shape is not None:
                    self.energy_l1_neuronwise[n][e_dir] = torch.zeros(shape)

    def after_step(self, state):
        for n, p in state['model'].named_parameters():
            prev_p = self.provider.previous_parameters[n].to(p.device)
            delta = (p - prev_p).abs()

            for e_dir in self.energy_direction:
                if e_dir not in self.energy_l1_neuronwise[n]:
                    continue
                
                energy = None
                if p.ndim <= 2: # For biases or simple 2D params
                    energy = delta
                elif p.ndim >= 3:
                    if e_dir == 'incoming':
                        energy = delta.sum(dim=tuple(range(2, p.ndim)))
                    elif e_dir == 'outgoing':
                        energy = delta.sum(dim=1)

                if energy is not None:
                    self.energy_l1_neuronwise[n][e_dir] += energy.detach().cpu()

    def after_test(self, state):
        for n, layer_energies in self.energy_l1_neuronwise.items():
            for e_dir, energy in layer_energies.items():
                state['data']['energies_l1_neuronwise'][n][e_dir].append(energy.clone().numpy())
        
class MinimumEnergyL2NetworkTracker(Interceptor):
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train(self, state):
        self.minimum_energy_l2 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l2'] = []
        
    def after_test(self, state):
        delta = torch.zeros((state['n_networks'])).to(state['device'])
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n] 
            delta += ((
                           (p - init_p)**2
                               ).sum(dim=tuple(range(1, p.ndim))
                                   )
                        )
        delta = delta ** 0.5
        self.minimum_energy_l2 = delta.detach().cpu()
        state['data']['minimum_energies_l2'].append(self.minimum_energy_l2.clone().numpy())   
        
class EnergyL2NetworkTracker(Interceptor):
    def __init__(self, previous_parameter_provider):
        self.provider = previous_parameter_provider
    
    def before_train(self, state):
        self.energy_l2 = torch.zeros((state['n_networks'],))
        state['data']['energies_l2'] = []
        
    def after_step(self, state):
        delta = torch.zeros((state['n_networks'])).to(state['device'])
        for n, p in state['model'].named_parameters():
            prev_p = self.provider.previous_parameters[n] 
            delta += ((
                           (p - prev_p)**2
                               ).sum(dim=tuple(range(1, p.ndim))
                                   )
                        )
        delta = delta ** 0.5
        self.energy_l2 += delta.detach().cpu()
        
    def after_test(self, state):
        state['data']['energies_l2'].append(self.energy_l2.clone().numpy())
        
class EnergyL0NetworkTracker(Interceptor):
    def __init__(self, previous_parameter_provider):
        self.provider = previous_parameter_provider
        
    def before_train(self, state):
        self.energy_l0 = torch.zeros((state['n_networks'],), dtype=torch.long)
        state['data']['energies_l0'] = []
        
    def after_step(self, state):
        for n, p in state['model'].named_parameters():
            prev_p = self.provider.previous_parameters[n]
            delta = torch.count_nonzero(p - prev_p, dim=(tuple(range(1, p.ndim))))
            self.energy_l0 += delta.detach().cpu()
    
    def after_test(self, state):
        state['data']['energies_l0'].append(self.energy_l0.clone().numpy())
        
class MinimumEnergyL0NetworkTracker(Interceptor):
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
        
    def before_train(self, state):
        self.minimum_energy_l0 = torch.zeros((state['n_networks'],), dtype=torch.long)
        state['data']['minimum_energies_l0'] = []
    
    def after_test(self, state):
        delta = torch.zeros((state['n_networks'],)).to(state['device'])
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n] 
            delta += torch.count_nonzero(p - init_p, dim=(tuple(range(1, p.ndim))))
        self.minimum_energy_l0 = delta.detach().cpu()
        state['data']['minimum_energies_l0'].append(self.minimum_energy_l0.clone().numpy())
        
########################### PARAMETER ITERATOR ################################

"""
ParameterIterator is designed to make Interceptors that require looping over
parameters more flexible and more efficient. We might make an
Interceptor for each data we're interested in. However, if these loop over
parameters then we make multiple calls for a loop. Instead, ParameterIterator
takes multiple of these Interceptors (termed Handlers) and calls them for each
parameter.

    for interceptor in interceptors                    for layer in layers
         for layer in interceptor.layers      vs.           interceptor1_func(layer)
             func(layer)                                    interceptor2_func(layer)
"""

class ParameterIterator(Interceptor):
    """
    An interceptor that efficiently loops over parameters and dispatches to
    handlers based on their registered events, preventing wasteful computation.
    """
    def __init__(self, handlers):
        super().__init__()
        self.handlers = handlers
        # Pre-compute which handlers listen to which events for max performance
        self.event_map = self._map_events(handlers)

    def _map_events(self, handlers):
        event_map = {}
        for handler in handlers:
            for event_name, funcs in handler.events.items():
                for func_type in funcs: # 'func' or 'log'
                    key = f"{event_name}_{func_type}"
                    if key not in event_map:
                        event_map[key] = []
                    event_map[key].append(handler)
        return event_map

    def _dispatch(self, state, event_key):
        if event_key not in self.event_map:
            return # No handlers are listening, do nothing.

        if event_key.endswith('_func'):
            for n, p in state['model'].named_parameters():
                for handler in self.event_map[event_key]:
                    getattr(handler, event_key)(n, p, state)
        else: # _log
            for handler in self.event_map[event_key]:
                getattr(handler, event_key)(state)

    def before_train(self, state): self._dispatch(state, 'before_train_log')
    def before_test(self, state): self._dispatch(state, 'before_test_log')
    def after_step(self, state):
        self._dispatch(state, 'after_step_func')
        self._dispatch(state, 'after_step_log')
    def after_test(self, state):
        self._dispatch(state, 'after_test_func')
        self._dispatch(state, 'after_test_log')
        
class ParameterIterator(Interceptor):
    def __init__(self, handlers):
        self.handlers = handlers
        
    def _func(self, state, event_name): # use when you want to compute over individual parameters
        event_name += '_func'
        for n, p in state['model'].named_parameters():
            for handler in self.handlers:
                if hasattr(handler, event_name):
                    #print(handler)
                    getattr(handler, event_name)(n, p, state) # (state, n, p) might be better syntax but alas
                    
    def _log(self, state, event_name): # used when only the state is necessary
        event_name += '_log'
        for handler in self.handlers:
            if hasattr(handler, event_name):
                getattr(handler, event_name)(state)
        
    def _should_run(self, signal, event_name):
        return any(
            signal in handler.events.get(event_name, [])
            for handler in self.handlers
        )
      
    def before_train(self, state):
        if self._should_run('func', 'before_train'):
            self._func(state, 'before_train')
        
        if self._should_run('log', 'before_train'):
            self._log(state, 'before_train')
        
    def before_update(self, state):
        if self._should_run('func', 'before_update'):
            self._func(state, 'before_update')
        
        if self._should_run('log', 'before_update'):
            self._log(state, 'before_update')
    
    def after_update(self, state):
        if self._should_run('func', 'after_update'):
            self._func(state, 'after_update')
        
        if self._should_run('log', 'after_update'):
            self._log(state, 'after_update')
    
    def before_step(self, state):
        if self._should_run('func', 'before_step'):
            self._func(state, 'before_step')
        
        if self._should_run('log', 'before_step'):
            self._log(state, 'before_step')
    
    def after_step(self, state):
        if self._should_run('func', 'after_step'):
            self._func(state, 'after_step')
        
        if self._should_run('log', 'after_step'):
            self._log(state, 'after_step')
            
    def before_test(self, state):
        if self._should_run('func', 'before_test'):
            self._func(state, 'before_test')
        
        if self._should_run('log', 'before_test'):
            self._log(state, 'before_test')
        
    def after_test(self, state):
        if self._should_run('func', 'after_test'):
            self._func(state, 'after_test')
        
        if self._should_run('log', 'after_test'):
            self._log(state, 'after_test')

########################### HANDLERS ################################
"""
Handlers "handle" parameters during a loop over parameters. They require a
[dict] with key events they intercept along with a value [list: 'log', 'func'] 
indicating the computation that happens: a function or log. Functions expect
name of parameter ('n'), parameter values ('p') and state.
"""

            
class EnergyL0NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'after_step' : ['func'],
            'after_test' : ['log']
        }
    
    def __init__(self, previous_parameter_provider):
        self.provider = previous_parameter_provider

    def before_train_log(self, state):
        self.energy_l0 = torch.zeros((state['n_networks'],), dtype=torch.long)
        state['data']['energies_l0'] = []
        

    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n]
        delta = torch.count_nonzero(p - prev_p, dim=(tuple(range(1, p.ndim))))
        self.energy_l0 += delta.detach().cpu()
    
    def after_test_log(self, state):
        state['data']['energies_l0'].append(self.energy_l0.clone().numpy())
        
class MinimumEnergyL0NetworkHandler(Handler):
    
        
    events = {
            'before_train' : ['log'],
            'before_test' : ['log'],
            'after_test' : ['func', 'log']
        }

    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
        
    def before_train_log(self, state):
        state['data']['minimum_energies_l0'] = []
    
    def before_test_log(self, state):
        self.minimum_energy_l0 = torch.zeros((state['n_networks'],)).to(state['device'])
        
    def after_test_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n] 
        self.minimum_energy_l0 += torch.count_nonzero(p - init_p, dim=(tuple(range(1, p.ndim))))
    
    def after_test_log(self, state):
        state['data']['minimum_energies_l0'].append(self.minimum_energy_l0.detach().cpu().clone().numpy())

class EnergyL1NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'after_step' : ['func'],
            'after_test' : ['log']
        }
    
    def __init__(self, previous_parameter_provider):
        self.provider = previous_parameter_provider
    
    def before_train_log(self, state):
        self.energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['energies_l1'] = []
    
    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n] 
        delta = (p - prev_p).abs().sum(dim=tuple(range(1, p.ndim)))
        self.energy_l1 += delta.detach().cpu()
        
    def after_test_log(self, state):
        state['data']['energies_l1'].append(self.energy_l1.clone().numpy())
        
class MinimumEnergyL1NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'before_test' : ['log'],
            'after_test' : ['func', 'log']
        }
    
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train_log(self, state):
        state['data']['minimum_energies_l1'] = []
    
    def before_test_log(self, state):
        self.minimum_energy_l1 = torch.zeros((state['n_networks'],)).to(state['device'])
        
    def after_test_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n] 
        self.minimum_energy_l1 += (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))
        
    def after_test_log(self, state):
        state['data']['minimum_energies_l1'].append(self.minimum_energy_l1.detach().cpu().clone().numpy())

class MinimumEnergyL1LayerwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'before_test': ['log'],
        'after_test': ['func', 'log']
    }
    
    def __init__(self, initial_parameter_provider):
        super().__init__()
        self.provider = initial_parameter_provider

    def before_train_log(self, state):
        state['data']['minimum_energies_l1_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
    
    def before_test_log(self, state):
        self.current_minimum_energy = {}
        
    def after_test_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n].to(p.device)
        delta = (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))
        self.current_minimum_energy[n] = delta.detach().cpu()

    def after_test_log(self, state):
        for n, energy in self.current_minimum_energy.items():
            state['data']['minimum_energies_l1_layerwise'][n].append(energy.clone().numpy())

class EnergyL1LayerwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
    }

    def __init__(self, previous_parameter_provider):
        super().__init__()
        self.provider = previous_parameter_provider

    def before_train_log(self, state):
        self.energy_l1_layerwise = {
            n: torch.zeros((state['n_networks'],)) for n, _ in state['model'].named_parameters()
        }
        state['data']['energies_l1_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
        
    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n].to(p.device)
        delta = (p - prev_p).abs().sum(dim=tuple(range(1, p.ndim)))
        self.energy_l1_layerwise[n] += delta.detach().cpu()
            
    def after_test_log(self, state):
        for n, energy in self.energy_l1_layerwise.items():
            state['data']['energies_l1_layerwise'][n].append(energy.clone().numpy())

class MinimumEnergyL1NeuronwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'before_test': ['log'],
        'after_test': ['func', 'log']
    }
    
    def __init__(self, initial_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = initial_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction
        
    def before_train_log(self, state):
        state['data']['minimum_energies_l1_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            state['data']['minimum_energies_l1_neuronwise'][n] = {
                e_dir: [] for e_dir in self.energy_direction
            }
                
    def before_test_log(self, state):
        self.current_minimum_energy = {}
        
    def after_test_func(self, n, p, state):
        self.current_minimum_energy[n] = {}
        init_p = self.provider.initial_parameters[n].to(p.device)
        delta = (p - init_p).abs()
        
        for e_dir in self.energy_direction:
            energy = None
            if p.ndim <= 2: energy = delta
            elif p.ndim >= 3:
                if e_dir == 'incoming': 
                    energy = delta.sum(dim=tuple(range(2, p.ndim)))
                elif e_dir == 'outgoing': 
                    energy = delta.sum(dim=1)
            
            if energy is not None:
                self.current_minimum_energy[n][e_dir] = energy.detach().cpu()
                
    def after_test_log(self, state):
        for n, layer_energies in self.current_minimum_energy.items():
            for e_dir, energy in layer_energies.items():
                state['data']['minimum_energies_l1_neuronwise'][n][e_dir].append(energy.clone().numpy())

class EnergyL1NeuronwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
    }

    def __init__(self, previous_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = previous_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction

    def before_train_log(self, state):
        self.energy_l1_neuronwise = {}
        state['data']['energies_l1_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            self.energy_l1_neuronwise[n] = {}
            state['data']['energies_l1_neuronwise'][n] = {e_dir: [] for e_dir in self.energy_direction}
            for e_dir in self.energy_direction:
                shape = None
                if p.ndim <= 2: shape = p.shape
                elif p.ndim >= 3:
                    if e_dir == 'incoming': 
                        shape = p.shape[:2]
                    elif e_dir == 'outgoing': 
                        shape = (p.shape[0],) + p.shape[2:]
                if shape is not None:
                    self.energy_l1_neuronwise[n][e_dir] = torch.zeros(shape)

    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n].to(p.device)
        delta = (p - prev_p).abs()
        for e_dir in self.energy_direction:
            if e_dir not in self.energy_l1_neuronwise[n]: 
                continue
            energy = None
            if p.ndim <= 2: 
                energy = delta
            elif p.ndim >= 3:
                if e_dir == 'incoming': 
                    energy = delta.sum(dim=tuple(range(2, p.ndim)))
                elif e_dir == 'outgoing': 
                    energy = delta.sum(dim=1)
            if energy is not None:
                self.energy_l1_neuronwise[n][e_dir] += energy.detach().cpu()

    def after_test_log(self, state):
        for n, layer_energies in self.energy_l1_neuronwise.items():
            for e_dir, energy in layer_energies.items():
                state['data']['energies_l1_neuronwise'][n][e_dir].append(energy.clone().numpy())
        
class MinimumEnergyL2NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'before_test' : ['log'],
            'after_test' : ['func', 'log']
        }
    
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train_log(self, state):
        self.minimum_energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l2'] = []
        
    def before_test_log(self, state):
        self.minimum_energy_l2 = torch.zeros((state['n_networks'],)).to(state['device'])
        
    def after_test_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n] 
        self.minimum_energy_l2 += ((
                        (p - init_p)**2
                             ).sum(dim=tuple(range(1, p.ndim))
                                 )
                        )
        
    def after_test_log(self, state):
        self.minimum_energy_l2 = self.minimum_energy_l2 ** 0.5
        state['data']['minimum_energies_l2'].append(self.minimum_energy_l2.detach().cpu().clone().numpy())
        
class EnergyL2NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'before_update' : ['log'],
            'after_step' : ['func'],
            'after_update' : ['log'],
            'after_test' : ['log']
        }
    
    def __init__(self, previous_parameter_provider):
        self.provider = previous_parameter_provider
    
    def before_train_log(self, state):
        self.energy_l2 = torch.zeros((state['n_networks'],))
        state['data']['energies_l2'] = []
        
    def before_update_log(self, state):
        self.delta = torch.zeros((state['n_networks'],)).to(state['device'])
    
    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n] 
        self.delta += ((
                    (p - prev_p)**2
                         ).sum(dim=tuple(range(1, p.ndim))
                             )
                    )
        
    def after_update_log(self, state):
        self.delta = self.delta ** 0.5
        self.energy_l2 += self.delta.detach().cpu()
        
    def after_test_log(self, state):
        state['data']['energies_l2'].append(self.energy_l2.clone().numpy())
        
class MinimumEnergyL2LayerwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'before_test': ['log'],
        'after_test': ['func', 'log']
    }
    
    def __init__(self, initial_parameter_provider):
        super().__init__()
        self.provider = initial_parameter_provider

    def before_train_log(self, state):
        state['data']['minimum_energies_l2_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
        
    def before_test_log(self, state):
        self.current_minimum_energy = {}
        
    def after_test_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n].to(p.device)
        delta = ((p - init_p)**2).sum(dim=tuple(range(1, p.ndim)))**0.5
        self.current_minimum_energy[n] = delta.detach().cpu()
    
    def after_test_log(self, state):
        for n, energy in self.current_minimum_energy.items():
            state['data']['minimum_energies_l2_layerwise'][n].append(energy.clone().numpy())

class EnergyL2LayerwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
    }

    def __init__(self, previous_parameter_provider):
        super().__init__()
        self.provider = previous_parameter_provider

    def before_train_log(self, state):
        self.energy_l2_layerwise = {
            n: torch.zeros((state['n_networks'],)) for n, _ in state['model'].named_parameters()
        }
        state['data']['energies_l2_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
        
    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n].to(p.device)
        delta = ((p - prev_p)**2).sum(dim=tuple(range(1, p.ndim)))**0.5
        self.energy_l2_layerwise[n] += delta.detach().cpu()
            
    def after_test_log(self, state):
        for n, energy in self.energy_l2_layerwise.items():
            state['data']['energies_l2_layerwise'][n].append(energy.clone().numpy())

class MinimumEnergyL2NeuronwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'before_test'  : ['log'],
        'after_test': ['func', 'log']
    }
    
    def __init__(self, initial_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = initial_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction
        
    def before_train_log(self, state):
        state['data']['minimum_energies_l2_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            state['data']['minimum_energies_l2_neuronwise'][n] = {
                e_dir: [] for e_dir in self.energy_direction
            }
                
    def before_test_log(self, state):
        self.current_minimum_energy = {}
    
    def after_test_func(self, n, p, state):
        self.current_minimum_energy[n] = {}
        init_p = self.provider.initial_parameters[n].to(p.device)
        delta_sq = (p - init_p)**2
        
        for e_dir in self.energy_direction:
            energy = None
            if p.ndim <= 2: 
                energy = delta_sq**0.5
            elif p.ndim >= 3:
                if e_dir == 'incoming': 
                    energy = delta_sq.sum(dim=tuple(range(2, p.ndim)))**0.5
                elif e_dir == 'outgoing': 
                    energy = delta_sq.sum(dim=1)**0.5
            
            if energy is not None:
                self.current_minimum_energy[n][e_dir] = energy.detach().cpu()
        
    def after_test_log(self, state):
        for n, layer_energies in self.current_minimum_energy.items():
            for e_dir, energy in layer_energies.items():
                state['data']['minimum_energies_l2_neuronwise'][n][e_dir].append(energy.clone().numpy())

class EnergyL2NeuronwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
    }

    def __init__(self, previous_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = previous_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction

    def before_train_log(self, state):
        self.energy_l2_neuronwise = {}
        state['data']['energies_l2_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            self.energy_l2_neuronwise[n] = {}
            state['data']['energies_l2_neuronwise'][n] = {e_dir: [] for e_dir in self.energy_direction}
            for e_dir in self.energy_direction:
                shape = None
                if p.ndim <= 2: 
                    shape = p.shape
                elif p.ndim >= 3:
                    if e_dir == 'incoming': 
                        shape = p.shape[:2]
                    elif e_dir == 'outgoing': 
                        shape = (p.shape[0],) + p.shape[2:]
                if shape is not None:
                    self.energy_l2_neuronwise[n][e_dir] = torch.zeros(shape)

    def after_step_func(self, n, p, state):
        prev_p = self.provider.previous_parameters[n].to(p.device)
        delta_sq = (p - prev_p)**2
        for e_dir in self.energy_direction:
            if e_dir not in self.energy_l2_neuronwise[n]: continue
            energy = None
            if p.ndim <= 2: 
                energy = delta_sq**0.5
            elif p.ndim >= 3:
                if e_dir == 'incoming': 
                    energy = delta_sq.sum(dim=tuple(range(2, p.ndim)))**0.5
                elif e_dir == 'outgoing': 
                    energy = delta_sq.sum(dim=1)**0.5
            if energy is not None:
                self.energy_l2_neuronwise[n][e_dir] += energy.detach().cpu()

    def after_test_log(self, state):
        for n, layer_energies in self.energy_l2_neuronwise.items():
            for e_dir, energy in layer_energies.items():
                state['data']['energies_l2_neuronwise'][n][e_dir].append(energy.clone().numpy())
