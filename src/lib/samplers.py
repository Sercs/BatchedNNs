import torch
import numpy as np
import math

from functools import partial

from torch.utils.data import Sampler

# TODO: shuffle and roll

def ensemble_collate_fn(batch, n_streams):
    if len(batch) % n_streams != 0:
        raise ValueError("Batch size is not divisible by the number of networks!")

    features = torch.stack([
                item[0] for item in batch
                ])
    labels = torch.stack([
                item[1] for item in batch
                ])
    
    indices = torch.tensor([item[2] for item in batch], dtype=torch.long)

    feature_dim = features.shape[-1]
    label_dim = labels.shape[-1]

    features = features.view(-1, n_streams, feature_dim)
    labels = labels.view(-1, n_streams, label_dim)
    indices = indices.view(-1, n_streams)
    
    return features, labels, indices

def collate_fn(n_streams):
    return partial(ensemble_collate_fn, n_streams=n_streams)

class IdenticalSampler(Sampler):
    """
    A PyTorch Sampler used to explicitly copy items in a batch such that all
    networks view exactly the same items at exactly the same time.
    """
    def __init__(self, data_source, n_streams, batch_size, drop_last=False):
        self.data_source = data_source
        self.n_streams = n_streams
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist()
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            final_batch = np.repeat(batch_indices, self.n_streams).tolist()
            yield final_batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return math.ceil(len(self.data_source) / self.batch_size)

class RandomSampler(Sampler):
    """
    A PyTorch Sampler used to explicitly randomize items in a batch such that all
    networks view different items at the same time.
    """
    def __init__(self, data_source, n_streams, batch_size, drop_last=False):
        self.data_source = data_source
        self.n_streams = n_streams
        self.batch_size = batch_size
        self.num_samples = len(self.data_source)
        self.drop_last = drop_last

    def __iter__(self):
        shuffled_indices_per_net = [
            torch.randperm(self.num_samples) for _ in range(self.n_streams)
        ]
        stacked_indices = torch.stack(shuffled_indices_per_net)
        transposed_indices = stacked_indices.T
        
        for i in range(0, self.num_samples, self.batch_size):
            chunk = transposed_indices[i : i + self.batch_size]

            if self.drop_last and chunk.shape[0] < self.batch_size:
                continue
            yield chunk.flatten().tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return math.ceil(self.num_samples / self.batch_size)

class VaryBatchAndDatasetSizeSampler(Sampler):
    """
    A PyTorch Sampler for training with multiple parallel data streams ("networks"),
    where each stream can have a different virtual dataset size and batch size,
    all drawing from a common data source.

    Args:
        data_source (Dataset): The dataset to sample from.
        n_streams (int): The number of parallel data streams.
        dataset_sizes (int or list[int]): The virtual number of unique samples
            each network sees in an epoch.
        batch_sizes (int or list[int]): The batch size for each network.
        padding_value (int, optional): Value to use for padding batches. Defaults to -1.
        drop_last (bool, optional): If True, drop the last incomplete batch. Defaults to True.
        order (str, optional): 'identical' or 'random'.
            - 'identical': Each stream sees the same samples, in the same order if possible.
            Different dataset sizes results in the same samples but different 
            ordering due to differences in the dataset size.
            - 'random': Each stream sees completely different data, ordered randomly.
            Defaults to 'identical'.
        method (str, optional): 'buffer', 'loop', or 'stretch'.
            - 'buffer': First come first serve, then idle till everyone finishes.
            i.e. (assuming padding_value=-1, batch_sizes=1, dataset_sizes=[1,2,4])
            [ # samples
                [1, -1, -1, -1] # stream 1 (note repeated -1 at the end of epoch)
                [1,  2, -1, -1] # stream 2
                [1,  2,  3,  4] # stream 3
            ]
            - 'loop': Creates a fixed pool of `dataset_size` and loops over it,
              reshuffling the pool only at the start of a new master epoch.
              i.e. (assuming padding_value=-1, batch_sizes=1, dataset_sizes=[1,2,4])
              [ # samples
                  [1,  1,  1,  1] # stream 1 (note: samples repeat)
                  [1,  2,  2,  1] # stream 2 (note: random shuffling)
                  [1,  2,  3,  4] # stream 3
              ]
            - 'stretch': Evenly distributes batches from faster networks across
              the duration of the slowest network's epoch.
              
              i.e. (assuming padding_value=-1, batch_sizes=1, dataset_sizes=[1,2,4])
              [ # samples
                  [-1, -1, -1, 1] # stream 1
                  [-1,  1, -1, 2] # stream 2 # (note: samples get evenly distributed)
                  [ 1,  2,  3, 4] # stream 3
              ]  
              
            In terms of 'buffer', with different dataset sizes, the smaller dataset 
            will pass through all samples and then continue doing nothing until the 
            largest dataset stream finishes. The intention was to syncronize epochs 
            across different dataset sizes but 'stretch' is likely the better option 
            since we intermittently test. With 'buffer' the end tests effectively do 
            nothing and most learning occurs on the first samples. The even 
            distribution of 'stretch' ensures that testing periods are roughly 
            equivalent over epochs. 'loop' sees much less issues but also has less
            control over the ordering of samples being seen and requires consideration
            when accounting for epochs or testing periods.
             
            Defaults to 'stretch'.
            
            In terms of batch sizes, we compute batches up to the largest batch_size
            and pad the rest. In other words, with batch_sizes=[1, 4], 
            dataset_sizes = [2, 8], padding_value=-1. {} indicates batch.
            [
                [{1, -1, -1, -1}, {2, -1, -1, -1}] # stream 1 (note how batches are padded)
                [{1,  2,  3,  4}, {5,  6,  7,  8}] # stream 2
            ]
            
            Given how quickly large batches can get through an epoch (i.e. 256x faster)
            there may be significantly wasted compute when using very different 
            batches.
    """
    def __init__(self, 
                 data_source, 
                 n_streams, 
                 dataset_sizes,
                 batch_sizes,
                 padding_value = -1, 
                 drop_last = True,
                 order = 'identical', # or 'random'
                 method = 'stretch'): # or 'loop' or 'buffer'
        
        if not isinstance(n_streams, int) or n_streams <= 0:
            raise ValueError("n_streams must be a positive integer.")
            
        if isinstance(dataset_sizes, int):
            dataset_sizes = [dataset_sizes] * n_streams
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes] * n_streams
        
        if len(dataset_sizes) != n_streams:
            raise ValueError("Length of dataset_sizes must equal n_streams.")
        if len(batch_sizes) != n_streams:
            raise ValueError("Length of batch_sizes must equal n_streams.")
        if order not in ['identical', 'random']:
            raise ValueError("order must be 'identical' or 'random'.")
        if method not in ['buffer', 'loop', 'stretch']:
            raise ValueError("method must be 'buffer', 'loop', or 'stretch'.")

        self.data_source = data_source
        self.num_samples = len(data_source)
        self.n_streams = n_streams
        self.dataset_sizes = torch.tensor(dataset_sizes, dtype=torch.long)
        self.batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)
        self.padding_value = padding_value
        self.drop_last = drop_last
        self.order = order
        self.method = method

        if self.dataset_sizes.max().item() > self.num_samples:
            raise ValueError("A dataset_size cannot be larger than the data_source.")
        if (self.batch_sizes < 0).any():
            raise ValueError("batch_sizes must not contain negative integers.")
        if (self.dataset_sizes < 0).any():
                raise ValueError("dataset_sizes must not contain negative integers.")

        self.max_batch_size = self.batch_sizes.max().item()

        self.master_perm = None
        self.sample_pools = None

        if self.order == 'identical':
            # Ccreate one master permutation for all epochs to draw from.
            # networks will get a prefix of this of size `dataset_size`.
            self.master_perm = torch.randperm(self.num_samples)
        else: # 'random'
            # create unique, persistent pools for each network. These pools are fixed
            # across epochs, and only their internal order is shuffled.
            self.sample_pools = []
            for i in range(self.n_streams):
                ds_size = self.dataset_sizes[i].item()
                # each network gets its own random subset, fixed across epochs
                pool = torch.randperm(self.num_samples)[:ds_size]
                self.sample_pools.append(pool)

    def get_samples_per_network(self):
        """
        Returns the fixed pool of sample indices for each network.

        This is useful for inspecting which data samples are allocated to which
        network for the entire duration of training. The order of samples within
        the pool will be shuffled at the start of each epoch, but the set of
        samples in the pool remains constant.
        """
        if self.order == 'identical':
            pools = []
            for i in range(self.n_streams):
                ds_size = self.dataset_sizes[i].item()
                pools.append(self.master_perm[:ds_size])
            return pools
        else: # 'random'
            return self.sample_pools

    def _get_num_batches(self):
        # create a safe version of batch_sizes to avoid division by zero.
        safe_batch_sizes = self.batch_sizes.float().clone()
        safe_batch_sizes[safe_batch_sizes == 0] = 1 

        if self.drop_last:
            # integer division (floor)
            num_batches = torch.floor(self.dataset_sizes.float() / safe_batch_sizes)
        else:
            # ceiling division
            num_batches = torch.ceil(self.dataset_sizes.float() / safe_batch_sizes)
        
        # invalidate networks with batch_size 0
        num_batches[self.batch_sizes == 0] = 0
        return num_batches.long()

    def __len__(self) -> int:
        # number of batches is determined by the network that runs for the longest.
        num_batches = self._get_num_batches()
        max_batches = num_batches.max().item() if len(num_batches) > 0 else 0
        return int(max_batches)

    def __iter__(self):
        if self.method == 'loop':
            if len(self) == 0:
                shuffled_indices = torch.empty((self.n_streams, 0), dtype=torch.long)
                stream_lengths = torch.zeros(self.n_streams, dtype=torch.long)
            else:
                total_samples_per_network = len(self) * self.batch_sizes
                full_stream_list = []

                if self.order == 'identical':
                    for i in range(self.n_streams):
                        ds_size = self.dataset_sizes[i].item()
                        total_samples_needed = total_samples_per_network[i].item()

                        if ds_size == 0 or total_samples_needed == 0:
                            full_stream_list.append(torch.empty(0, dtype=torch.long))
                            continue
                        
                        # the fixed pool for this network is a prefix of the master permutation.
                        sample_pool = self.master_perm[:ds_size]
                        num_epochs_local = (total_samples_needed + ds_size - 1) // ds_size

                        epoch_parts = []
                        for _ in range(num_epochs_local):
                            # re-shuffle the fixed pool independently for each local epoch.
                            shuffled_pool = sample_pool[torch.randperm(len(sample_pool))]
                            epoch_parts.append(shuffled_pool)
                        
                        full_stream = torch.cat(epoch_parts)
                        full_stream_list.append(full_stream[:total_samples_needed])

                else: # 'random' order
                    for i in range(self.n_streams):
                        ds_size = self.dataset_sizes[i].item()
                        total_samples_needed = total_samples_per_network[i].item()

                        if ds_size == 0 or total_samples_needed == 0:
                            full_stream_list.append(torch.empty(0, dtype=torch.long))
                            continue
                        
                        # the pool is a unique random subset for this network, fixed across epochs.
                        sample_pool = self.sample_pools[i]
                        num_epochs_local = (total_samples_needed + ds_size - 1) // ds_size

                        epoch_parts = []
                        for _ in range(num_epochs_local):
                            # re-shuffle the unique pool for each local epoch.
                            shuffled_pool = sample_pool[torch.randperm(len(sample_pool))]
                            epoch_parts.append(shuffled_pool)
                        
                        full_stream = torch.cat(epoch_parts)
                        full_stream_list.append(full_stream[:total_samples_needed])

                stream_lengths = torch.tensor([len(s) for s in full_stream_list], dtype=torch.long)
                max_len = stream_lengths.max().item() if len(stream_lengths) > 0 else 0
                padded = [torch.nn.functional.pad(s, (0, max_len - len(s)), value=self.padding_value) for s in full_stream_list]
                shuffled_indices = torch.stack(padded) if padded else torch.empty((self.n_streams, 0), dtype=torch.long)
        
        else: # 'buffer' or 'stretch' method
            # create shuffled streams from the persistent pools for this epoch.
            shuffled_streams = []
            if self.order == 'identical':
                # create one consistent shuffle for the epoch based on the largest dataset
                max_ds_size = self.dataset_sizes.max().item() if self.n_streams > 0 else 0
                largest_pool = self.master_perm[:max_ds_size]
                epoch_shuffled_base = largest_pool[torch.randperm(len(largest_pool))]

                for i in range(self.n_streams):
                    ds_size = self.dataset_sizes[i].item()
                    # each network gets a prefix of the single shuffled base pool
                    shuffled_streams.append(epoch_shuffled_base[:ds_size])
            else: # 'random'
                for i in range(self.n_streams):
                    # the pool is pre-defined and fixed for this network.
                    sample_pool = self.sample_pools[i]
                    # shuffle the pool for this epoch's iteration.
                    shuffled_streams.append(sample_pool[torch.randperm(len(sample_pool))])
            
            # pad streams to the max dataset size so they can be stacked.
            max_ds_size = self.dataset_sizes.max().item() if self.n_streams > 0 else 0
            
            padded_streams = [
                torch.nn.functional.pad(s, (0, max_ds_size - len(s)), value=self.padding_value)
                for s in shuffled_streams
            ]
            shuffled_indices = torch.stack(padded_streams) if padded_streams else torch.empty((self.n_streams, 0), dtype=torch.long)

        if self.method == 'stretch':
            num_batches_per_network = self._get_num_batches()
            max_batches = len(self)

            if max_batches == 0:
                yield from ()
                return

            stretch_accumulators = torch.zeros(self.n_streams, dtype=torch.long)
            positions = torch.zeros(self.n_streams, dtype=torch.long)
            
            for _ in range(max_batches):
                step_batch = torch.full((self.n_streams, self.max_batch_size), self.padding_value, dtype=torch.long)
                
                stretch_accumulators += num_batches_per_network
                fire_mask = stretch_accumulators >= max_batches
                
                for i in range(self.n_streams):
                    if fire_mask[i]:
                        batch_size = self.batch_sizes[i].item()
                        if batch_size > 0:
                            start = positions[i].item()
                            end = start + batch_size
                            # the pool for this network has size ds_size
                            ds_size = self.dataset_sizes[i].item()
                            # ensure we don't go out of bounds on the source pool
                            clamped_end = min(end, ds_size)
                            indices_to_take = torch.arange(start, clamped_end)
                            
                            if indices_to_take.numel() > 0:
                                batch_data = torch.gather(shuffled_indices[i], 0, indices_to_take)
                                step_batch[i, :len(batch_data)] = batch_data

                            positions[i] += batch_size
                
                stretch_accumulators[fire_mask] -= max_batches

                yield step_batch.T.flatten().tolist()
            return

        # create a batch range tensor (for 'buffer' and 'loop')
        positions = torch.zeros(self.n_streams, dtype=torch.long)
        batch_range = torch.arange(self.max_batch_size)

        for _ in range(len(self)):
            indices_to_get = positions.unsqueeze(1) + batch_range.unsqueeze(0)
            valid_batch_mask = batch_range.unsqueeze(0) < self.batch_sizes.unsqueeze(1)
            
            if self.method == 'buffer':
                in_dataset_mask = indices_to_get < self.dataset_sizes.unsqueeze(1)
                valid_mask = valid_batch_mask & in_dataset_mask
                
                # clamp indices to prevent out-of-bounds access to the shuffled list
                max_len = shuffled_indices.shape[1]
                clamped_indices = indices_to_get.clamp(max=max_len - 1) if max_len > 0 else indices_to_get
                step_batch = torch.gather(shuffled_indices, 1, clamped_indices)
                
                step_batch[~valid_mask] = self.padding_value
            
            else: # 'loop' method
                max_len = shuffled_indices.shape[1]
                clamped_indices = indices_to_get.clamp(max=max_len - 1) if max_len > 0 else indices_to_get
                
                step_batch = torch.gather(shuffled_indices, 1, clamped_indices) if max_len > 0 else torch.full_like(clamped_indices, self.padding_value)
                
                in_stream_mask = indices_to_get < stream_lengths.unsqueeze(1)
                valid_mask = valid_batch_mask & in_stream_mask
                step_batch[~valid_mask] = self.padding_value

            positions += self.batch_sizes
            
            yield step_batch.T.flatten().tolist()

class FixedEpochSampler(Sampler):
    """
    A PyTorch Sampler designed specifically for evaluation loops on batched networks
    where each network "views" a differently sized dataset and thus requires masking
    samples for networks who "view" a smaller dataset. This sampler guarantees a 
    1 epoch over mixed datasets.

    This sampler takes a pre-defined list of sample indices for each network stream
    and generates batches using a fixed batch size. It handles uneven dataset
    sizes by padding streams that finish early, ensuring all networks are processed
    for the same number of steps. This is ideal for use with a TestLoop interceptor.

    Args:
        data_source (Dataset): The dataset to sample from (required by Sampler).
        indices_per_network (list[torch.Tensor]): A list where each element is a
            tensor of sample indices for one network stream.
            Intended use with VaryBatchAndDatasetSizeSampler(...).get_samples_per_network().
        batch_size (int): The fixed batch size to use for all networks.
        padding_value (int, optional): The value to use for padding. Defaults to -1.
    """
    def __init__(self, data_source, indices_per_network, batch_size, padding_value=-1):
        super().__init__()
        
        if not isinstance(indices_per_network, list):
            raise TypeError("indices_per_network must be a list of tensors.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.data_source = data_source
        
        if isinstance(indices_per_network, np.ndarray):
            indices_per_network = [torch.tensor(n) for n in indices_per_network]
        self.indices_per_network = indices_per_network
        self.batch_size = batch_size
        self.padding_value = padding_value
        
        self.n_streams = len(self.indices_per_network)
        self.dataset_sizes = torch.tensor([len(indices) for indices in self.indices_per_network])
        
        # Calculate the number of batches needed for the longest stream
        num_batches_per_network = torch.ceil(self.dataset_sizes.float() / self.batch_size).long()
        self._len = num_batches_per_network.max().item() if self.n_streams > 0 else 0

    def __len__(self) -> int:
        """The number of batches is determined by the longest-running network."""
        return self._len

    def __iter__(self):
        """
        Yields batches of indices. Each batch is a flattened list of indices
        of size (batch_size * n_streams), structured as if it came from a
        (batch_size, n_streams) tensor.
        """
        # Pointers to track the current position in each network's index list
        positions = torch.zeros(self.n_streams, dtype=torch.long)

        for _ in range(self._len):
            # Create a placeholder for the batch for all streams
            # Shape: (n_streams, batch_size)
            step_batch = torch.full((self.n_streams, self.batch_size), self.padding_value, dtype=torch.long)
            
            for i in range(self.n_streams):
                start = positions[i].item()
                end = start + self.batch_size
                
                # Get the slice of indices for the current network
                # Clamp the end to avoid going out of bounds
                indices_slice = self.indices_per_network[i][start:end]
                if len(indices_slice) > 0:
                    step_batch[i, :len(indices_slice)] = indices_slice

            # Update positions for the next iteration
            positions += self.batch_size
            
            # The DataLoader expects a flattened list of indices for one batch.
            # We transpose to group by sample index across networks, then flatten.
            # Shape (batch_size, n_streams) -> flattened list
            yield step_batch.T.flatten().tolist()

# note that using num_workers > 0 lags when hard_samples are put into the batch
# the lag is greater the more workers. This is negligible in most cases as it 
# only effects the first few forward passes of training.
class HardMiningSampler(Sampler):
    """
    A PyTorch Sampler for batched networks that performs hard negative mining
    by querying a hard index provider, using a fixed total batch size but
    allowing for variable ratios of hard-to-normal samples per stream.
    """
    def __init__(self, data_source, n_streams, hard_index_provider, batch_size,
                 hard_samples_per_batch, padding_value=-1):
        super().__init__()

        # --- Validation ---
        if not hasattr(hard_index_provider, 'get_hard_indices'):
            raise TypeError("hard_index_provider must have a 'get_hard_indices' method.")
        
        # --- Store Configuration ---
        self.data_source = data_source
        self.n_streams = n_streams
        self.hard_index_provider = hard_index_provider
        self.padding_value = padding_value
        self.batch_size = batch_size
        self.num_samples = len(data_source)

        # --- Per-stream setup ---
        self.hard_samples_per_batch = self._init_param(hard_samples_per_batch, "hard_samples_per_batch")
        self.normal_samples_per_batch = self.batch_size - self.hard_samples_per_batch
        
        if (self.normal_samples_per_batch < 1).any():
            raise ValueError(f"n_hard_samples must leave at least 1 normal sample within the batch" + 
                             f" got batch_size={batch_size} and n_hard_samples={hard_samples_per_batch}" +
                             f"leaving {self.normal_samples_per_batch}.")

        # --- State for each epoch ---
        self.normal_indices_queues = []
        self._len = self.num_samples // batch_size

    def _init_param(self, param, name):
        """Helper to convert int inputs to a tensor."""
        if isinstance(param, int):
            param = [param] * self.n_streams
        if len(param) != self.n_streams:
            raise ValueError(f"Length of {name} must equal n_streams.")
        return torch.tensor(param, dtype=torch.long)

    def _prepare_epoch_queues(self):
        """Pre-shuffles all indices for the epoch's 'normal' queue."""
        self.normal_indices_queues = []
        num_batches_per_stream = []

        all_indices = list(range(len(self.data_source)))
        
        for i in range(self.n_streams):
            shuffled_all_indices = all_indices.copy()
            np.random.shuffle(shuffled_all_indices)
            self.normal_indices_queues.append(shuffled_all_indices)
            
            num_normal = len(shuffled_all_indices)
            normal_per_batch = self.normal_samples_per_batch[i].item()

            num_batches_per_stream.append(num_normal // normal_per_batch)
        
        self._len = max(num_batches_per_stream) if num_batches_per_stream else 0

    def __len__(self):
        return self._len

    def __iter__(self):

        shuffled_indices_per_net = [
            torch.randperm(self.num_samples) for _ in range(self.n_streams)
        ]
        transposed_indices = torch.stack(shuffled_indices_per_net).T
        for batch_idx in range(self._len):

            batch_indices = torch.full((self.n_streams, self.batch_size), self.padding_value, dtype=torch.long)
            
            for stream_idx in range(self.n_streams):

                num_normal_needed = self.normal_samples_per_batch[stream_idx].item()
                start = batch_idx * num_normal_needed
                end = start + num_normal_needed

                batch_normal = transposed_indices[start:end, stream_idx].tolist()
                

                num_hard_needed = self.hard_samples_per_batch[stream_idx].item()
                batch_hard = []
                if num_hard_needed > 0:
                    current_hard_list = self.hard_index_provider.get_hard_indices(stream_idx)
                    if len(current_hard_list) > 0:
                        replace = len(current_hard_list) < num_hard_needed
                        batch_hard = np.random.choice(
                            current_hard_list, size=num_hard_needed, replace=replace
                        ).tolist()

                final_stream_batch = batch_normal + batch_hard
                np.random.shuffle(final_stream_batch)
                if final_stream_batch:
                    batch_indices[stream_idx, :len(final_stream_batch)] = torch.tensor(final_stream_batch)
            
            yield batch_indices.T.flatten().tolist()


                
            
        
            
        
        
        
        