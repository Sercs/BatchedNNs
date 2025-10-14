import torch
import torch.nn as nn
import numpy as np

class BatchOptimizer(torch.optim.Optimizer):
    """
    An abstract base class for PyTorch optimizers with enhanced support for
    batch-wise hyperparameter broadcasting and modular update steps.
    
    This optimizer is designed to handle hyperparameters (e.g., learning rate,
    momentum) that are provided as tensors. It automatically processes and reshapes
    these tensors to be broadcastable with the model's parameters. This enables
    applying different hyperparameter values to different examples within the same
    batch. Useful for hyperparameter grid searches.

    The optimization step is broken down into distinct methods:
        - `_get_updates_for_param`: Must be implemented by subclasses to compute
          the update vector (e.g., `-lr * grad`).
        - `_apply_updates`: Applies the computed update to the parameter.
        
      This separation is crucial. It allows wrapper optimizers, like the `Competitive` 
      optimizer, to intercept the calculated update vector from `_get_updates_for_param`, 
      modify it (e.g., by masking it), and then apply the final result. It can
      also intercept states within the optimizer like momentum.
    
    Subclasses (e.g., `SGD`, `AdamW`) must implement the `_get_updates_for_param` method 
    to define their specific optimization logic.
    """
    def __init__(self, params, defaults):
        for name, val in defaults.items():
            defaults[name] = self._process_input(val, name)
            
        super().__init__(params, defaults)
        
    def _process_input(self, val, name):
        if isinstance(val, (float, list, np.ndarray)):
            val = torch.tensor(val, dtype=torch.float32)
        if isinstance(val, torch.Tensor):
            if torch.any(val < 0.0):
                raise ValueError(f"Invalid {name} value found in tensor: {val}")
            return val
        raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")
        
    def _prepare_params(self, p, param_state, group):
        for name, val in group.items():
            if name in 'params': continue # no need to prepare parameters
            if isinstance(val, torch.Tensor):  # using * for unwrapping and multiplying... absolute cinema
                prepared_val = val.to(p.device).view(-1, *[1] * (p.ndim - 1))
            else:
                prepared_val = val
            param_state[f'{name}_prepared'] = prepared_val
            
    def _get_updates_for_param(self, p, param_state, group):
        raise NotImplementedError # not implemented in the main optimizer, must be implemented in subclass
    
    @torch.no_grad()
    def step(self, closure=None):
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                if 'lr_prepared' not in param_state: # all optimizers require a lr
                    self._prepare_params(p, param_state, group)
                    
                update = self._get_updates_for_param(p, param_state, group) 
                
                self._apply_updates(p, update, param_state, group)
                
        if closure is not None:
            with torch.enable_grad():
                return closure()
            
    def _apply_updates(self, p, update, param_state, group):
        p.data.add_(update)

class Competitive(torch.optim.Optimizer):
    """
    A wrapper optimizer that applies competitive updates by masking a fraction of the
    smallest updates (gradients, momentum, etc.) for each parameter.

    This optimizer supports various configurations for the masking fraction 'k',
    allowing for fine-grained control across batches, layers, and specific parameters.

    Args:
        optimizer (torch.optim.Optimizer): The base optimizer (e.g., Adam, SGD).
        k (float, list, np.ndarray, torch.Tensor): The fraction of updates to mask.
            - float: A single fraction applied to all weight parameters.
            - 1D Tensor: A fraction for each network in the batch, applied across all layers.
            - 2D Tensor (per-weight): Fractions for each weight layer and each network.
            - 2D Tensor (per-parameter): Fractions for each parameter (incl. biases) and network.
        selection_key (str): The key in the optimizer's state to use for selection.
            Defaults to 'update', but can be 'grad' or a momentum key like 'exp_avg'.
    """
    def __init__(self, optimizer, k, selection_key='update', 
                                     competition_mode='layer_wise_weight',
                                     neuron_competition_dim=None,
                                     bias_competition=False):
        
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer must be a torch.optim.Optimizer, not {type(optimizer)}")

        self.optimizer = optimizer
        self.selection_key = selection_key
        self.bias_competition = bias_competition


        self.valid_modes = ['layer_wise_weight', 'neuron_wise_weight', 'layer_wise_neuron']
        if competition_mode not in self.valid_modes:
            raise ValueError(f"competition_mode must be one of {self.valid_modes}, but got '{competition_mode}'")
        self.competition_mode = competition_mode

        is_neuron_related_mode = 'neuron' in self.competition_mode
        if is_neuron_related_mode:
            if neuron_competition_dim is None:
                raise ValueError(
                    "neuron_competition_dim must be 'incoming' or 'outgoing' "
                    f"for competition_mode '{self.competition_mode}'."
                )
            if neuron_competition_dim not in ['incoming', 'outgoing']:
                raise ValueError("neuron_competition_dim must be 'incoming' or 'outgoing'")
        else: # it's 'layer_wise_weight'
            if neuron_competition_dim is not None:
                print(f"neuron_competition_dim='{neuron_competition_dim}' was provided but will be ignored "
                    "for mode 'layer_wise_weight'.")
                
        self.neuron_competition_dim = neuron_competition_dim

        defaults = optimizer.defaults.copy()
        defaults['mask_fraction'] = self._process_input(k, 'k')
        super().__init__(self.optimizer.param_groups, defaults)

        self._param_to_indices = {}
        self._all_params_list = []
        self._is_weight_param = []
        param_idx_counter = 0
        weight_idx_counter = 0

        for group in self.param_groups:
            for p in group['params']:
                self._all_params_list.append(p)
                
                is_weight = p.dim() > 2 
                self._is_weight_param.append(is_weight)

                self._param_to_indices[p] = {
                    'param_idx': param_idx_counter,
                    'weight_idx': weight_idx_counter if is_weight else None
                }
                param_idx_counter += 1
                if is_weight:
                    weight_idx_counter += 1
        
        self.num_total_params = len(self._all_params_list)
        self.num_weight_params = sum(self._is_weight_param)


    def _process_input(self, val, name):
        """Validates that k is a valid fraction between 0.0 and 1.0."""
        if isinstance(val, (float, list, np.ndarray)):
            val = torch.tensor(val, dtype=torch.float32)
        if isinstance(val, torch.Tensor):
            if torch.any((val < 0.0) | (val > 1.0)):
                raise ValueError(f"All values for {name} (mask_fraction) must be between 0.0 and 1.0.")
            return val
        raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

    def _prepare_params(self, p, competitive_state, group):
        batch_size = p.shape[0] if p.dim() > 0 else 1
        device = p.device
        k = group['mask_fraction'].to(device)
        k_per_param = None
    
        is_bias = p.dim() <= 1
        if is_bias and not self.bias_competition:
            # for biases without competition, force a dense update by setting k=1.0.
            k_per_param = torch.ones(batch_size, device=device, dtype=torch.float32)
        else:
            # handle all k shapes as before.
            if k.ndim == 0:  # k = scalar -> apply same competition to all nets
                k_per_param = k.expand(batch_size)
            elif k.ndim == 1:  # k = 1D tensor -> apply each competition to each net
                if len(k) != batch_size:
                    raise ValueError(f"1D k length ({len(k)}) must match batch size ({batch_size}).")
                k_per_param = k
            elif k.ndim == 2:  # k = 2D tensor tensor -> apply each competition to each net and each layer
                num_k_rows = k.shape[0]
                if self.bias_competition:
                    if num_k_rows != self.num_total_params:
                        raise ValueError(f"With bias_competition=True, 2D k rows ({num_k_rows}) must match total params ({self.num_total_params}).")
                    param_idx = self._param_to_indices[p]['param_idx']
                    k_per_param = k[param_idx]
                else: # p must be a weight here
                    if num_k_rows != self.num_weight_params:
                        raise ValueError(f"With bias_competition=False, 2D k rows ({num_k_rows}) must match weight params ({self.num_weight_params}).")
                    weight_idx = self._param_to_indices[p]['weight_idx']
                    k_per_param = k[weight_idx]
            else:
                raise ValueError(f"k must be a scalar, 1D, or 2D tensor, but has {k.ndim} dimensions.")
    
        num_elements = p[0].numel() if p.dim() > 1 else p.shape[1] if p.dim() > 1 else (p.shape[0] if p.dim() == 1 else 1)
        num_w_to_keep = torch.ceil(k_per_param * num_elements).long()
        num_to_keep_clamped = torch.clamp(num_w_to_keep, 0, num_elements)
        competitive_state['num_w_to_keep'] = num_to_keep_clamped

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                # get the state from the base optimizer
                param_state = self.optimizer.state[p]
                if 'lr_prepared' not in param_state: # all optimizers require a lr
                    self.optimizer._prepare_params(p, param_state, group)
                
                # the competitive optimizer has its own state, primarily for the masked_k (or num_w_to_keep since masking in)
                competitive_state = self.state[p]
                if 'num_w_to_keep' not in competitive_state:
                    self._prepare_params(p, competitive_state, group)
                #print(p.dim())
                is_bias = p.dim() <= 2

                if is_bias and not self.bias_competition:
                    # Apply a standard, non-competitive update and skip
                    param_state = self.optimizer.state[p]
                    update = self.optimizer._get_updates_for_param(p, param_state, group)
                    self._apply_updates(p, update, param_state, group)
                    continue

                update = self.optimizer._get_updates_for_param(p, param_state, group)
                selection = self._get_selection_tensor(p, update, param_state, group)
                
                if is_bias:
                    mask = self._generate_bias_mask(p, selection, competitive_state)
                else:
                    mask = self._generate_competition_mask(p, selection, competitive_state)
            
                update.mul_(mask)
                
                self._apply_updates(p, update, param_state, group)
                
    def _generate_competition_mask(self, p, selection, competitive_state):
        """
        Generates the competition mask based on the optimizer's configuration.
        This new helper function contains the core logic for all competition modes.
        """
        num_w_to_keep = competitive_state['num_w_to_keep']
        
        if self.competition_mode == 'neuron_wise_weight':

            batch_size, out_features, in_features = p.shape
            selection_abs = selection.abs()
        
            if self.neuron_competition_dim == 'incoming':
                selection_abs = selection_abs.transpose(1, 2)
        
            num_neurons_in_batch, num_synapses_per_neuron = selection_abs.shape[1], selection_abs.shape[2]
            selection_flat = selection_abs.reshape(-1, num_synapses_per_neuron)
        
            total_weights = out_features * in_features
            k_per_batch = num_w_to_keep.float() / total_weights
            
            # ensure each neuron within each linear has the same competition level
            # arr = [1, 2] (technically torch.Tensor)
            # arr.repeat_interleave(3) -> arr == [1, 1, 1, 2, 2, 2]
            k_expanded = k_per_batch.repeat_interleave(num_neurons_in_batch)
            num_synapses_to_keep = torch.ceil(k_expanded * num_synapses_per_neuron).long()

            # make sure k is valid
            num_synapses_to_keep.clamp_(max=num_synapses_per_neuron)
        
            _, sorted_indices = torch.sort(selection_flat, dim=1, descending=True)
            
            # create a boolean mask for the top k values in the sorted tensor
            arange_mask = torch.arange(num_synapses_per_neuron, device=p.device).expand_as(selection_flat)
            k_mask = arange_mask < num_synapses_to_keep.unsqueeze(1)

            mask_flat = torch.zeros_like(selection_flat, dtype=torch.bool)
            mask_flat.scatter_(1, sorted_indices, k_mask) # pull from k_mask at sorted indices along dim=1
            
            mask = mask_flat.view(batch_size, num_neurons_in_batch, num_synapses_per_neuron)
            if self.neuron_competition_dim == 'incoming':
                mask = mask.transpose(1, 2)
        
            return mask.float()
        

        elif self.competition_mode == 'layer_wise_neuron':
            # 'outgoing' (axonal): sum over `in_features`. Demand shape: (n, out_features)
            # 'incoming' (dendritic): sum over `out_features`. Demand shape: (n, in_features)
            sum_dim = 2 if self.neuron_competition_dim == 'outgoing' else 1
            neuron_demand = selection.abs().sum(dim=sum_dim) # Shape: (n, num_neurons)
            
            num_neurons = neuron_demand.shape[1]
            
            total_weights = p[0].numel()
            k_per_batch = num_w_to_keep.float() / total_weights
            num_neurons_to_keep = torch.ceil(k_per_batch * num_neurons).long()
            
            sorted_demand, _ = torch.sort(neuron_demand, dim=1, descending=True)
            thresholds = torch.full((selection.shape[0],), float('inf'), device=p.device, dtype=p.dtype)
            
            mask_needed = num_neurons_to_keep > 0
            if mask_needed.any():
                indices = (num_neurons_to_keep[mask_needed] - 1).unsqueeze(1)
                gathered_thresholds = torch.gather(sorted_demand[mask_needed], 1, indices).squeeze(1)
                thresholds[mask_needed] = gathered_thresholds

            thresholds_reshaped = thresholds.view(-1, *[1] * (neuron_demand.ndim - 1))
            neuron_mask = neuron_demand >= thresholds_reshaped # Shape: (n, num_neurons)
            
            # expand neuron mask to the full weight tensor shape
            mask = neuron_mask.unsqueeze(sum_dim) # Add back the summed dimension
            return mask.expand_as(selection)
    

        else: # self.competition_mode == 'layer_wise_weight'
            selection_flat = selection.abs().view(selection.shape[0], -1)
            num_elements = selection_flat.shape[1]
            num_to_keep = num_w_to_keep
        
            _, sorted_indices = torch.sort(selection_flat, dim=1, descending=True)
        
            arange_mask = torch.arange(num_elements, device=p.device).expand_as(selection_flat)
            k_mask = arange_mask < num_to_keep.unsqueeze(1)
        
            mask_flat = torch.zeros_like(selection_flat, dtype=torch.bool)
            mask_flat.scatter_(1, sorted_indices, k_mask)
            
            mask = mask_flat.view_as(selection)
            return mask.float()
        
    def _generate_bias_mask(self, p, selection, competitive_state):
        """
        Generates a competition mask for a bias parameter.
    
        This performs a simple top-k selection on the elements of the bias vector
        for each network in the batch.
        """
        num_to_keep = competitive_state['num_w_to_keep']
        selection_abs = selection.abs()
        
        sorted_vals, _ = torch.sort(selection_abs, dim=1, descending=True)
        
        thresholds = torch.full((selection.shape[0],), float('inf'), device=p.device, dtype=p.dtype)
        
        # only calculate thresholds for batches where we need to drop something
        mask_needed = num_to_keep > 0
        if mask_needed.any():
            max_elements = selection.shape[1]
            # ensure valid
            indices = (num_to_keep[mask_needed].clamp(max=max_elements) - 1).long().unsqueeze(1)
            
            gathered_thresholds = torch.gather(sorted_vals[mask_needed], 1, indices).squeeze(1)
            thresholds[mask_needed] = gathered_thresholds
            
        thresholds_reshaped = thresholds.view(-1, 1)
        mask = selection_abs >= thresholds_reshaped
        return mask.float()

    def _get_selection_tensor(self, p, update, param_state, group):
        """Determines which tensor to use for ranking and masking."""
        if self.selection_key == 'update':
            return update
        elif self.selection_key == 'grad':
            return p.grad
        
        # get optimizer buffer
        if self.selection_key in param_state:
            return param_state[self.selection_key] # note: param_state comes from the wrapped optimizer 
        else:
            raise KeyError(
                f"Selection key '{self.selection_key}' not found in the base optimizer's state. "
                f"Available keys: {list(param_state.keys())}. Or use 'update' or 'grad'."
            )
            
    def _apply_updates(self, p, update, param_state, group):
        p.data.add_(update)
    
class SGD(BatchOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
     
    def _prepare_params(self, p, param_state, group):
        super()._prepare_params(p, param_state, group)
        momentum = param_state.get('momentum_prepared', 0.0)
        param_state['use_momentum'] = (isinstance(momentum, float) and momentum > 0.0) or \
                                     (isinstance(momentum, torch.Tensor) and (momentum > 0.0).any())

        if param_state['use_momentum'] and 'momentum_buffer' not in param_state:
            param_state['momentum_buffer'] = torch.zeros_like(p).detach()
            
    def _get_updates_for_param(self, p, param_state, group):
        lr = param_state['lr_prepared']
        grad = p.grad
        
        grad_update = grad # make the update agnostic to momentum or not
        
        if param_state['use_momentum']:
            momentum = param_state['momentum_prepared']
            buf = param_state['momentum_buffer']

            buf.mul_(momentum).add_(grad)
            grad_update = buf

        # note: we return the full step vector (scaled by -lr) so wrappers can modify it directly.
        return grad_update * -lr
    
    # inherits apply update

class AdamW(BatchOptimizer):
    """
    Implements a refactored AdamW optimizer subclassing BatchOptimizer.

    Supports broadcasting of learning rate, betas, and weight decay
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    Epsilon (eps) is treated as a scalar float.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        weight_decay=weight_decay, eps=eps)

        super().__init__(params, defaults)

    def _prepare_params(self, p, param_state, group):
        super()._prepare_params(p, param_state, group)

        if 'step' not in param_state:
            param_state['step'] = 0
            param_state['exp_avg'] = torch.zeros_like(p).detach()
            param_state['exp_avg_sq'] = torch.zeros_like(p).detach()

    def _get_updates_for_param(self, p, param_state, group):
        self._prepare_params(p, param_state, group)

        exp_avg = param_state['exp_avg']
        exp_avg_sq = param_state['exp_avg_sq']
        grad = p.grad
        
        lr = param_state['lr_prepared']
        beta1 = param_state['beta1_prepared']
        beta2 = param_state['beta2_prepared']
        weight_decay = param_state['weight_decay_prepared']
        eps = group['eps']

        param_state['step'] += 1
        step = param_state['step']

        if not (isinstance(weight_decay, float) and weight_decay == 0.0):
            p.data.mul_(1.0 - lr * weight_decay)

        exp_avg.mul_(beta1).addcmul_(grad, (1 - beta1), value=1.0)
        exp_avg_sq.mul_(beta2).addcmul_(grad.pow(2), (1.0 - beta2), value=1.0)

        bias_correction1 = 1.0 - beta1.pow(step)
        bias_correction2 = 1.0 - beta2.pow(step)

        denom = exp_avg_sq.sqrt().div_(bias_correction2.sqrt_()).add_(eps) # Using sqrt for stability
        update = exp_avg.div_(denom).mul_(-lr).div_(bias_correction1)
        return update

    # inherits apply update

class AdamP(BatchOptimizer):
    """
    Implements a refactored AdamP-like optimizer (Generalized Adam) 
    subclassing BatchOptimizer.

    Supports broadcasting of learning rate, betas, degree, and weight decay.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, degree=2.0,
                 weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        degree=degree, weight_decay=weight_decay, eps=eps)

        super().__init__(params, defaults)

    def _prepare_params(self, p, param_state, group):
        super()._prepare_params(p, param_state, group)

        if 'step' not in param_state:
            param_state['step'] = 0
            param_state['exp_avg'] = torch.zeros_like(p).detach()
            param_state['exp_avg_sq'] = torch.zeros_like(p).detach()

    def _get_updates_for_param(self, p, param_state, group):
        self._prepare_params(p, param_state, group)

        exp_avg = param_state['exp_avg']
        exp_avg_sq = param_state['exp_avg_sq']
        grad = p.grad
        
        lr = param_state['lr_prepared']
        beta1 = param_state['beta1_prepared']
        beta2 = param_state['beta2_prepared']
        weight_decay = param_state['weight_decay_prepared']
        degree = param_state['degree_prepared']
        eps = group['eps']

        param_state['step'] += 1
        step = param_state['step']

        if not (isinstance(weight_decay, float) and weight_decay == 0.0):
            p.data.mul_(1.0 - lr * weight_decay)


        exp_avg.mul_(beta1).addcmul_(grad, (1 - beta1), value=1.0)

        grad_pow = torch.pow(grad.abs(), degree)
        exp_avg_sq.mul_(beta2).addcmul_(grad_pow, (1.0 - beta2), value=1.0) 

        bias_correction1 = 1.0 - beta1.pow(step)
        bias_correction2 = 1.0 - beta2.pow(step)

        denom = exp_avg_sq.div(bias_correction2).pow(1.0 / degree).add_(eps)
        update = exp_avg.div_(denom).mul_(-lr).div_(bias_correction1)
        return update
    
    # inherits apply update

class LazyAdamW(BatchOptimizer):
    """
    Implements a Lazy AdamW optimizer subclassing BatchOptimizer.

    Applies updates only to parameters with non-zero gradients in the current batch.
    Supports broadcasting of learning rate, betas, and weight decay.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        weight_decay=weight_decay, eps=eps)

        super().__init__(params, defaults)

    def _prepare_params(self, p, param_state, group):
        super()._prepare_params(p, param_state, group)

        if 'step' not in param_state:
            param_state['step'] = 0
            param_state['exp_avg'] = torch.zeros_like(p).detach()
            param_state['exp_avg_sq'] = torch.zeros_like(p).detach()

    def _get_updates_for_param(self, p, param_state, group):
        self._prepare_params(p, param_state, group)

        exp_avg = param_state['exp_avg']
        exp_avg_sq = param_state['exp_avg_sq']
        grad = p.grad
        

        lr = param_state['lr_prepared']
        beta1 = param_state['beta1_prepared']
        beta2 = param_state['beta2_prepared']
        weight_decay = param_state['weight_decay_prepared']
        eps = group['eps']

        param_state['step'] += 1
        step = param_state['step']

        update_needed_mask = (grad.sum(dim=tuple(range(1, p.ndim))) != 0).view(-1, *[1] * (p.ndim - 1)).float()
        
        if not (isinstance(weight_decay, float) and weight_decay == 0.0):
            wd_term = 1.0 - lr * weight_decay * update_needed_mask
            p.data.mul_(wd_term)

        exp_avg.mul_(beta1).addcmul_(grad, (1 - beta1), value=1.0)
        exp_avg_sq.mul_(beta2).addcmul_(grad.pow(2), (1.0 - beta2), value=1.0) 

        bias_correction1 = 1.0 - beta1.pow(step)
        bias_correction2 = 1.0 - beta2.pow(step)

        denom = exp_avg_sq.sqrt().div_(bias_correction2.sqrt_()).add_(eps)
        update = exp_avg.div_(denom).mul_(-lr).div_(bias_correction1)
        update.mul_(update_needed_mask)
        return update

class LazySGD(BatchOptimizer):
    """
    Implements a Lazy SGD optimizer subclassing BatchOptimizer.

    Applies updates only to parameters with non-zero gradients in the current batch.
    Supports broadcasting of learning rates and momentum values.
    """
    def __init__(self, params, lr, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def _prepare_params(self, p, param_state, group):
        super()._prepare_params(p, param_state, group)
        momentum = param_state.get('momentum_prepared', 0.0)
        param_state['use_momentum'] = (isinstance(momentum, float) and momentum > 0.0) or \
                                     (isinstance(momentum, torch.Tensor) and (momentum > 0.0).any())
        print(param_state['use_momentum'])

        if param_state['use_momentum'] and 'momentum_buffer' not in param_state:
            param_state['momentum_buffer'] = torch.zeros_like(p).detach()
        
        
    def _get_updates_for_param(self, p, param_state, group):
        lr = param_state['lr_prepared']
        grad = p.grad
        
        grad_update = grad # make the update agnostic to momentum or not
        
        if param_state['use_momentum']:
            momentum = param_state['momentum_prepared']
            buf = param_state['momentum_buffer']

            buf.mul_(momentum).add_(grad)
            grad_update = buf

        update_needed_mask = (grad.sum(dim=tuple(range(1, p.ndim))) != 0).view(-1, *[1] * (p.ndim - 1)).float()
        grad_update.mul_(update_needed_mask)

        return grad_update * -lr


        


        
        
            
            
            