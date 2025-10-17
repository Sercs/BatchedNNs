import torch
import numpy as np

class MaskComposer:
    """
    Builds a complex mask by composing or refining masks using a fluent interface.
    """
    def __init__(self, n_linears, n_in, n_out):
        """Initializes the composer with a fixed shape."""
        self.n_linears = n_linears
        self.n_out = n_out
        self.n_in = n_in
        
        self.weight_mask = None
        self.bias_mask = None

    def _get_shape_args(self):
        """Helper to get the stored shape as a dictionary."""
        return {'n_linears': self.n_linears, 'n_out': self.n_out, 'n_in': self.n_in}

    def start_with(self, mask_generator, **kwargs):
        """Starts a composition, generating the initial mask."""
        if not callable(mask_generator):
            config = mask_generator
            # Optional: Validate shape of pre-made mask
            if config['weight_mask'].shape != (self.n_linears, self.n_out, self.n_in):
                raise ValueError("Shape of provided mask config does not match composer's shape.")
        else:
            # Call the generator with the composer's stored shape
            func_args = {**self._get_shape_args(), **kwargs}
            config = mask_generator(**func_args)

        self.weight_mask = config['weight_mask'].clone()
        self.bias_mask = config['bias_mask'].clone()
        return self

    def _get_current_config(self):
        """Helper to package the current state into a config dict."""
        return {'weight_mask': self.weight_mask, 'bias_mask': self.bias_mask}

    def refine(self, masking_function, **kwargs):
        """
        Modifies the current mask by applying a function to it (replacement).
        """
        if self.weight_mask is None:
            raise RuntimeError("Cannot use .refine() before starting. Use .start_with() first.")
        
        # Generate a new config by refining the current one
        new_config = masking_function(base_mask_config=self._get_current_config(), **kwargs)
        
        # Replace the composer's state with the refined masks
        self.weight_mask = new_config['weight_mask']
        self.bias_mask = new_config['bias_mask']
        return self

    def intersect(self, mask_generator, **kwargs):
        """Applies a new mask using intersection (logical AND)."""
        if self.weight_mask is None:
            raise RuntimeError("Cannot use .intersect() before starting. Use .start_with() first.")
        
        if isinstance(mask_generator, MaskComposer):
            new_mask_config = mask_generator.get_mask()
        elif callable(mask_generator):
            func_args = {**self._get_shape_args(), **kwargs}
            new_mask_config = mask_generator(**func_args)
        else:
            new_mask_config = mask_generator

        self.weight_mask *= new_mask_config['weight_mask']
        self.bias_mask *= new_mask_config['bias_mask']
        return self

    def union(self, mask_generator, **kwargs):
        """Applies a new mask using union (logical OR)."""
        if self.weight_mask is None:
            raise RuntimeError("Cannot use .union() before starting. Use .start_with() first.")

        if isinstance(mask_generator, MaskComposer):
            new_mask_config = mask_generator.get_mask()
        elif callable(mask_generator):
            func_args = {**self._get_shape_args(), **kwargs}
            new_mask_config = mask_generator(**func_args)
        else:
            new_mask_config = mask_generator
            
        self.weight_mask = torch.clamp(self.weight_mask + new_mask_config['weight_mask'], 0, 1)
        self.bias_mask = torch.clamp(self.bias_mask + new_mask_config['bias_mask'], 0, 1)
        return self

    def get_mask(self, mask_activities=True, mask_gradients=True, bias_mode='from_weights', bias_density=None):
        """Returns the final composed masks as a config dictionary and resets."""
        if self.weight_mask is None:
            raise RuntimeError("Cannot get config before starting a composition.")
        
        mask_config = {
            'weight_mask': self.weight_mask,
            'bias_mask': _create_bias_mask(self.weight_mask, bias_mode, bias_density),
            'mask_activities': mask_activities,
            'mask_gradients': mask_gradients
        }
        self.weight_mask, self.bias_mask = None, None
        return mask_config

def _create_bias_mask(weight_mask, bias_mode, bias_density=None):
    """Helper to generate a bias mask based on the final weight mask."""
    n_linears, n_out, _ = weight_mask.shape
    device = weight_mask.device
    
    if bias_mode == 'on':
        return torch.ones(n_linears, n_out, device=device)
    elif bias_mode == 'off':
        return torch.zeros(n_linears, n_out, device=device)
    elif bias_mode == 'from_weights':
        # Activate bias if the neuron has any incoming connections
        return weight_mask.any(dim=-1).float()
    elif bias_mode == 'random':
        # Activate bias randomly based on a density
        if bias_density is None:
            raise Exception(f"Using bias_mode: {bias_mode} but bias_density not set, got {bias_density}")
        return (torch.rand(n_linears, n_out, device=device) < bias_density).float()
    else:
        raise ValueError(f"Unknown bias_mode: {bias_mode}")

def create_neuron_selection_mask(n_linears=None, n_out=None, n_in=None,
                                 density=0.1, 
                                 bias_mode='from_weights', 
                                 device='cpu', 
                                 base_mask_config=None,
                                 **kwargs):
    """Selects a random subset of entire neurons to be active."""
    if base_mask_config is None:
        if n_linears is None or n_out is None or n_in is None:
            raise ValueError("Shape must be provided for creation.")
        # Start with all neurons being active
        weight_mask = torch.ones((n_linears, n_out, n_in), device=device)
    else:
        weight_mask = base_mask_config['weight_mask'].clone()
        n_linears, n_out, n_in = weight_mask.shape # infer shape

    n_linears, n_out, n_in = weight_mask.shape
    
    if not isinstance(density, (list, np.ndarray, torch.Tensor)):
        density = [density]*n_linears
    
    for i in range(n_linears):
        # Find which neurons are currently active to select from
        active_indices = torch.where(weight_mask[i].any(dim=1))[0]
        n_active = len(active_indices)
        if n_active == 0: continue

        n_to_keep = max(1, int(density[i] * n_active)) if density[i] > 0 else 0
        perm = torch.randperm(n_active, device=device)
        indices_to_prune = active_indices[perm[n_to_keep:]]

        if len(indices_to_prune) > 0:
            weight_mask[i, indices_to_prune, :] = 0
            
    bias_mask = _create_bias_mask(weight_mask, bias_mode)
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask}


def create_global_sparsity_mask(n_linears=None, n_out=None, n_in=None,
                                density=0.1, 
                                bias_mode='from_weights', 
                                device='cpu', 
                                base_mask_config=None, 
                                **kwargs):
    """Creates layer-level sparsity by pruning individual connections globally."""
    if base_mask_config is None:
        if n_linears is None or n_out is None or n_in is None:
            raise ValueError("Shape must be provided for creation.")
        weight_mask = torch.ones((n_linears, n_out, n_in), device=device)
    else:
        weight_mask = base_mask_config['weight_mask'].clone()
        n_linears, n_out, n_in = weight_mask.shape # infer shape

    n_linears, n_out, n_in = weight_mask.shape
    
    if not isinstance(density, (list, np.ndarray, torch.Tensor)):
        density = [density]*n_linears

    for i in range(n_linears):
        active_indices = torch.where(weight_mask[i] == 1)
        n_active = len(active_indices[0])
        if n_active == 0: continue
        
        n_to_keep = max(1, int(density[i] * n_active)) if density[i] > 0 else 0
        
        perm = torch.randperm(n_active, device=device)
        indices_to_prune = (active_indices[0][perm[n_to_keep:]], active_indices[1][perm[n_to_keep:]])
        
        weight_mask[i][indices_to_prune] = 0
            
    bias_mask = _create_bias_mask(weight_mask, bias_mode)
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask}


def create_local_sparsity_mask(n_linears=None, n_out=None, n_in=None,
                               density=0.1,
                               bias_mode='from_weights', 
                               device='cpu', 
                               base_mask_config=None,
                               **kwargs):
    """Creates per-neuron sparsity by pruning connections for each neuron individually."""
    
    if not isinstance(density, (list, np.ndarray, torch.Tensor)):
        density = [density]*n_linears
        
    if base_mask_config is None:
        if n_linears is None or n_out is None or n_in is None:
            raise ValueError("Shape must be provided for creation.")
        weight_mask = torch.zeros((n_linears, n_out, n_in), device=device)
        
        for i in range(n_linears):
            n_to_keep_per_neuron = max(1, int(density[i] * n_in)) if density[i] > 0 else 0
            for j in range(n_out):
                perm = torch.randperm(n_in, device=device)
                indices_to_keep = perm[:n_to_keep_per_neuron]
                weight_mask[i, j, indices_to_keep] = 1.0
    else:
        weight_mask = base_mask_config['weight_mask'].clone()
        n_linears, n_out, n_in = weight_mask.shape # infer shape
        for i in range(n_linears):
            for j in range(n_out):
                active_indices = torch.where(weight_mask[i, j, :] == 1)[0]
                n_active = len(active_indices)
                if n_active == 0: continue

                n_to_keep = max(1, int(density[i] * n_active)) if density[i] > 0 else 0
                perm = torch.randperm(n_active, device=device)
                indices_to_prune = active_indices[perm[n_to_keep:]]
                
                if len(indices_to_prune) > 0:
                    weight_mask[i, j, indices_to_prune] = 0
    
    bias_mask = _create_bias_mask(weight_mask, bias_mode)
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask}

def create_local_connectivity_mask(n_linears=None, n_out=None,
                                     image_w=None, image_h=None,
                                     kernel_sizes=None, kernel_counts=None,
                                     bias_mode='from_weights', device='cpu',
                                     base_mask_config=None,
                                     n_in=None, # image_w*image_h gives n_in so here for compatibility with MaskComposer
                                     **kwargs):
    """
    Creates or refines a mask to enforce local receptive fields, like a CNN's first layer.
    Supports 1D specs (same for all networks) and 2D specs (unique for each network).
    """
    if image_w is None or image_h is None or kernel_sizes is None:
        raise ValueError("image_w, image_h, and kernel_sizes must be provided.")

    inferred_n_in = image_w * image_h
    if n_in is None:
        n_in = inferred_n_in
    elif n_in != inferred_n_in:
        raise ValueError(f"n_in ({n_in}) does not match image_w*image_h ({inferred_n_in}).")

    if base_mask_config is None:
        if n_linears is None or n_out is None:
            raise ValueError("n_linears and n_out must be provided for creation.")
    else:
        n_linears, n_out, _ = base_mask_config['weight_mask'].shape

    local_weight_mask = torch.zeros((n_linears, n_out, n_in), device=device)

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes]
    
    neuron_kernel_sizes = []
    is_2d_spec = isinstance(kernel_sizes[0], (list, np.ndarray, torch.Tensor))

    if is_2d_spec:
        if len(kernel_sizes) != n_linears:
            raise ValueError(f"2D spec's outer dimension ({len(kernel_sizes)}) must match n_linears ({n_linears}).")
        
        is_2d_counts = kernel_counts and isinstance(kernel_counts[0], (list, np.ndarray, torch.Tensor))
        if kernel_counts and is_2d_counts and len(kernel_counts) != n_linears:
            raise ValueError("2D kernel_counts spec must match n_linears dimension.")
            
        for i in range(n_linears):
            current_ks = kernel_sizes[i]
            current_kc = kernel_counts[i] if is_2d_counts else None
            
            neuron_kernel_sizes_1d = []
            if current_kc is None: # assume even distribution
                kernel_count = n_out // len(current_ks)
                for ks in current_ks:
                    neuron_kernel_sizes_1d.extend([ks] * kernel_count)
                remainder = n_out % len(current_ks)
                if remainder > 0: # fill the rest with the first kernel size
                    neuron_kernel_sizes_1d.extend([current_ks[0]] * remainder)
            else:
                if len(current_ks) != len(current_kc):
                    raise ValueError(f"Spec for linear {i} has mismatched lengths for kernel_sizes and kernel_counts.")
                for ks, kc in zip(current_ks, current_kc):
                    neuron_kernel_sizes_1d.extend([ks] * int(kc))
            neuron_kernel_sizes.append(neuron_kernel_sizes_1d)
    else:
        neuron_kernel_sizes_1d = []
        if kernel_counts is None: # similar logic to the 2D, except we don't need to loop over n_linear
            kernel_count = n_out // len(kernel_sizes)
            for ks in kernel_sizes:
                neuron_kernel_sizes_1d.extend([ks] * kernel_count)
            remainder = n_out % len(kernel_sizes)
            if remainder > 0:
                neuron_kernel_sizes_1d.extend([kernel_sizes[0]] * remainder)
        else:
            if len(kernel_sizes) != len(kernel_counts):
                raise ValueError("1D spec has mismatched lengths for kernel_sizes and kernel_counts.")
            for ks, kc in zip(kernel_sizes, kernel_counts):
                neuron_kernel_sizes_1d.extend([ks] * int(kc))
        neuron_kernel_sizes = [neuron_kernel_sizes_1d] * n_linears

    for i in range(n_linears):
        for j in range(n_out):
            kernel_size = neuron_kernel_sizes[i][j]
            
            # get a kernel
            y_offsets, x_offsets = torch.meshgrid(torch.arange(kernel_size, device=device), 
                                                  torch.arange(kernel_size, device=device), 
                                                  indexing='ij')
            kernel_offsets = torch.stack([y_offsets.flatten(), x_offsets.flatten()], dim=1)
            
            # find valid location
            x_max = image_w - kernel_size
            y_max = image_h - kernel_size
            anchor = torch.stack([
                torch.randint(0, y_max + 1, size=(1,), device=device), 
                torch.randint(0, x_max + 1, size=(1,), device=device)
            ], dim=1)
            
            # put it there
            kernel_idxs = anchor + kernel_offsets
            
            # shape it over (flat) mlp dims
            flat_idxs = kernel_idxs[:, 0] * image_w + kernel_idxs[:, 1]
            local_weight_mask[i, j, flat_idxs.long()] = 1.0

    if base_mask_config is None:
        weight_mask = local_weight_mask
    else:
        weight_mask = base_mask_config['weight_mask'] * local_weight_mask

    bias_mask = _create_bias_mask(weight_mask, bias_mode, **kwargs)
    
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask}

################## MASK STARTERS ################# 
# These don't interface with the MaskComposer and instead serve as a base.
def create_vary_width_masks(n_linears, n_ins, n_outs, device='cpu'):
    """Creates binary masks to simulate varying network widths."""
    if isinstance(n_ins, int): n_ins = [n_ins] * n_linears
    if isinstance(n_outs, int): n_outs = [n_outs] * n_linears
    
    max_in, max_out = max(n_ins), max(n_outs)
        
    weight_mask = torch.zeros((n_linears, max_out, max_in), device=device)
    bias_mask = torch.zeros((n_linears, max_out), device=device)
    
    for i in range(n_linears):
        n_active_in, n_active_out = n_ins[i], n_outs[i]
        weight_mask[i, :n_active_out, :n_active_in] = 1.0
        bias_mask[i, :n_active_out] = 1.0
        
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask, 'mask_gradients' : True, 'mask_activities' : True}

def create_subnetwork_mask(
    n_linears,
    layer_sizes,
    subnetwork_specs,
    device='cpu'
):
    """
    Creates masks for fully-connected subnetworks with randomly chosen neurons.

    Args:
        n_linears (int): The number of parallel subnetworks to create.
        layer_sizes (list[int]): Architecture of the full network, e.g., [784, 512, 256, 10].
        subnetwork_specs (list): Size of each hidden layer's subnetwork. The spec for each
                                 layer can be an integer or a float.
            - int >= 1: Absolute number of active neurons.
            - 0.0 < float <= 1.0: Fraction of active neurons.
                                 Can be specified in two ways:
            - 1D list ([256, 0.5]): Defined over hidden layers; applies the same spec to all `n_linears`.
            - 2D list ([[...], [...]]): Defined with shape (n_hidden_layers, n_linears),
              allowing a unique spec for each linear in each hidden layer.
        device (str, optional): The device to create tensors on. Defaults to 'cpu'.

    Returns:
        tuple[dict]: A tuple of mask_config dictionaries, allowing for direct unpacking.
    """
    n_hidden_layers = len(layer_sizes) - 2
    if n_hidden_layers == 0 and subnetwork_specs:
        raise ValueError("subnetwork_specs should be empty for a network with no hidden layers.")
    if n_hidden_layers > 0 and not subnetwork_specs:
        raise ValueError("subnetwork_specs must be provided for a network with hidden layers.")
        
    # --- 1. Parse the subnetwork_specs (Updated Logic) ---
    is_2d_spec = n_hidden_layers > 0 and isinstance(subnetwork_specs[0], (list, np.ndarray, torch.Tensor))
    if is_2d_spec:
        if len(subnetwork_specs) != n_hidden_layers:
            raise ValueError(f"2D spec's outer dimension ({len(subnetwork_specs)}) must match n_hidden_layers ({n_hidden_layers}).")
        if len(subnetwork_specs[0]) != n_linears:
            raise ValueError(f"2D spec's inner dimension ({len(subnetwork_specs[0])}) must match n_linears ({n_linears}).")
    elif n_hidden_layers > 0 and len(subnetwork_specs) != n_hidden_layers:
        raise ValueError(f"1D spec's length ({len(subnetwork_specs)}) must match n_hidden_layers ({n_hidden_layers}).")

    all_configs = [{} for _ in range(len(layer_sizes) - 1)]
    weight_masks = [torch.zeros((n_linears, layer_sizes[i+1], layer_sizes[i]), device=device) for i in range(len(layer_sizes)-1)]
    bias_masks = [torch.zeros((n_linears, layer_sizes[i+1]), device=device) for i in range(len(layer_sizes)-1)]

    for j in range(n_linears):
        prev_active_neurons = torch.ones(layer_sizes[0], dtype=torch.bool, device=device)
        for i in range(len(layer_sizes) - 1):
            is_hidden = i < n_hidden_layers
            if is_hidden:

                spec = subnetwork_specs[i][j] if is_2d_spec else subnetwork_specs[i]
                max_neurons = layer_sizes[i+1]
                
                if isinstance(spec, (int, float, np.integer)):
                    if spec >= 1: n_to_keep = int(spec)
                    elif 0.0 < spec < 1.0: n_to_keep = max(1, int(spec * max_neurons))
                    else: raise TypeError(f"Invalid spec value '{spec}'.")
                else: raise TypeError(f"Invalid spec type '{type(spec)}'.")
                
                if n_to_keep > max_neurons:
                    raise ValueError(f"Subnetwork size {n_to_keep} cannot exceed layer size {max_neurons}.")

                indices = torch.randperm(max_neurons)[:n_to_keep]
                current_active_neurons = torch.zeros(max_neurons, dtype=torch.bool, device=device)
                current_active_neurons[indices] = True
            else:
                current_active_neurons = torch.ones(layer_sizes[i+1], dtype=torch.bool, device=device)

            weight_mask_slice = current_active_neurons.unsqueeze(1) * prev_active_neurons.unsqueeze(0)
            weight_masks[i][j] = weight_mask_slice
            bias_masks[i][j] = current_active_neurons
            prev_active_neurons = current_active_neurons

    for i in range(len(all_configs)):
        all_configs[i] = {
            'weight_mask': weight_masks[i],
            'bias_mask': bias_masks[i],
            'mask_activities': True,
            'mask_gradients': True
        }

    return tuple(all_configs)

def all_on(n_linears, n_out, n_in, device='cpu', **kwargs):
    """Creates a fully connected mask of all ones."""
    weight_mask = torch.ones((n_linears, n_out, n_in), device=device)
    # The bias mask is also set to ones, but will be overridden by get_config
    bias_mask = torch.ones((n_linears, n_out), device=device)
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask}

def all_off(n_linears, n_out, n_in, device='cpu', **kwargs):
    """Creates a fully disconnected mask of all zeros."""
    weight_mask = torch.zeros((n_linears, n_out, n_in), device=device)
    bias_mask = torch.zeros((n_linears, n_out), device=device)
    return {'weight_mask': weight_mask, 'bias_mask': bias_mask}