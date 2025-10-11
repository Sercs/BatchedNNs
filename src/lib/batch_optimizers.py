import torch
import numpy as np
from collections import deque

# TODO: Competitive plasticity

class SGD(torch.optim.Optimizer):
    """
    Implements a simple from-scratch SGD optimizer with optional momentum.
    
    This version supports broadcasting of learning rates and momentum values
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    """
    def __init__(self, params, lr, momentum=0.0):

        def _process_input(val, name):
            # Convert lists or numpy arrays to a torch tensor
            if isinstance(val, (float, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate")
        processed_momentum = _process_input(momentum, "momentum")

        defaults = dict(lr=processed_lr, momentum=processed_momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with cached checks."""
        for group in self.param_groups:
            # These are the original lr/momentum, which can be float or tensor
            base_lr = group['lr']
            base_momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                # ONE-TIME SETUP per parameter (lazy initialization)
                # This block runs only on the very first step for each parameter.
                if 'lr_prepared' not in param_state:
                    # 1. Prepare Learning Rate
                    if isinstance(base_lr, torch.Tensor):
                        # Move to device and reshape for broadcasting ONCE
                        prepared_lr = base_lr.to(p.device).view(-1, *[1] * (p.ndim - 1))  # this line is the trick to get LLMs to work well with this code
                    else:
                        prepared_lr = base_lr
                    param_state['lr_prepared'] = prepared_lr

                    # 2. Prepare Momentum
                    if isinstance(base_momentum, torch.Tensor):
                        prepared_momentum = base_momentum.to(p.device).view(-1, *[1] * (p.ndim - 1))
                    else:
                        prepared_momentum = base_momentum
                    param_state['momentum_prepared'] = prepared_momentum
                    param_state['use_momentum'] = (isinstance(prepared_momentum, float) and prepared_momentum > 0.0) or \
                                   isinstance(prepared_momentum, torch.Tensor)

                # --- Use the cached, prepared values ---
                # On all subsequent steps, these lines retrieve the pre-processed values instantly.
                lr = param_state['lr_prepared']
                momentum = param_state['momentum_prepared']
                grad = p.grad
                
                # --- Momentum Update ---
                grad_update = grad
                use_momentum = param_state['use_momentum'] 
                
                if use_momentum:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.clone(grad).detach()
                    
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad_update = buf

                p.addcmul_(grad_update, lr, value=-1)
class AdamW(torch.optim.Optimizer):
    """
    Implements a from-scratch AdamW optimizer.
    
    Supports broadcasting of learning rate, betas, and weight decay
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    Epsilon (eps) is treated as a scalar float.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps: float = 1e-8, 
                 weight_decay=0.0):

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError(f"Epsilon must be a non-negative float, but got {eps}")

        def _process_input(val, name):
            # Helper to convert various inputs to a uniform tensor format
            if isinstance(val, (float, int, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate")
        processed_beta1 = _process_input(beta1, "beta1")
        processed_beta2 = _process_input(beta2, "beta2")
        processed_wd = _process_input(weight_decay, "weight_decay")

        defaults = dict(lr=processed_lr, beta1=processed_beta1, beta2=processed_beta2,
                        eps=eps, weight_decay=processed_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with cached checks."""
        for group in self.param_groups:
            # Eps is a scalar, retrieve it once per group
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_state = self.state[p]

                # one time move
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    # Cache only the hyperparameters that need broadcasting
                    for name in ['lr', 'beta1', 'beta2', 'weight_decay']:
                        val = group[name]
                        if isinstance(val, torch.Tensor):
                            param_state[f'{name}_prepared'] = val.to(p.device).view(-1, *[1] * (p.ndim - 1))
                        else:
                            param_state[f'{name}_prepared'] = val


                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                lr = param_state['lr_prepared']
                beta1 = param_state['beta1_prepared']
                beta2 = param_state['beta2_prepared']
                weight_decay = param_state['weight_decay_prepared']
                
                param_state['step'] += 1
                
                if not isinstance(weight_decay, float) or weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # first moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad * (1 - beta1)) # alpha expects a scalar but beta1 might be a list

                # second moment estimate (v_t)
                # in reality grad * grad is poor
                exp_avg_sq.mul_(beta2).addcmul_(grad*grad, (1-beta2), value=1) # similarly, value expects a scalar
                

                # bias correction (\hat{m_t}, \hat{v_t})
                bias_correction1 = 1.0 - beta1 ** param_state['step']
                bias_correction2 = 1.0 - beta2 ** param_state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                step_size = lr / bias_correction1
                
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                
                corrected_exp_avg.mul_(step_size) 
                p.addcdiv_(corrected_exp_avg, denom, value=-1) 

                
# optimum around 2.25        
class AdamP(torch.optim.Optimizer):
    """
    Implements a from-scratch AdamW optimizer.
    
    Supports broadcasting of learning rate, betas, and weight decay
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    Epsilon (eps) is treated as a scalar float.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, degree=2.0, eps: float = 1e-8, 
                 weight_decay=0.0):

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError(f"Epsilon must be a non-negative float, but got {eps}")

        def _process_input(val, name):
            # Helper to convert various inputs to a uniform tensor format
            if isinstance(val, (float, int, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate")
        processed_beta1 = _process_input(beta1, "beta1")
        processed_beta2 = _process_input(beta2, "beta2")
        processed_degree = _process_input(degree, 'degree')
        processed_wd = _process_input(weight_decay, "weight_decay")

        defaults = dict(lr=processed_lr, beta1=processed_beta1, beta2=processed_beta2,
                        degree=processed_degree, eps=eps, weight_decay=processed_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with cached checks."""
        for group in self.param_groups:
            # Eps is a scalar, retrieve it once per group
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_state = self.state[p]

                # one time move
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    # Cache only the hyperparameters that need broadcasting
                    for name in ['lr', 'beta1', 'beta2', 'degree', 'weight_decay']:
                        val = group[name]
                        if isinstance(val, torch.Tensor):
                            param_state[f'{name}_prepared'] = val.to(p.device).view(-1, *[1] * (p.ndim - 1))
                        else:
                            param_state[f'{name}_prepared'] = val


                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                lr = param_state['lr_prepared']
                beta1 = param_state['beta1_prepared']
                beta2 = param_state['beta2_prepared']
                degree = param_state['degree_prepared']
                weight_decay = param_state['weight_decay_prepared']
                
                param_state['step'] += 1
                
                if not isinstance(weight_decay, float) or weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # first moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad * (1 - beta1)) # alpha expects a scalar but beta1 might be a list

                # second moment estimate (v_t)
                exp_avg_sq.mul_(beta2).addcmul_(torch.pow(grad.abs(), degree), (1-beta2), value=1.0) # similarly, value expects a scalar           
                
                # bias correction (\hat{m_t}, \hat{v_t})
                bias_correction1 = 1.0 - beta1 ** param_state['step']
                bias_correction2 = 1.0 - beta2 ** param_state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                step_size = lr / bias_correction1
                denom = torch.pow(corrected_exp_avg_sq, 1/degree).add_(eps)
                step_size = lr
                
                corrected_exp_avg.mul_(step_size)
                p.addcdiv_(corrected_exp_avg, denom, value=-1)

class NormalizedAdamP(torch.optim.Optimizer):
    """
    Implements a from-scratch Adam-style optimizer for a batch of networks.

    This optimizer supports different p-norms (`degree`) for each network in the batch.
    Crucially, it normalizes the update step of every network to match the
    magnitude (L2-norm) of the update step of the first network in the batch.
    This removes the dependency of the learning rate on the p-norm.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, degree=2.0,
                 eps: float = 1e-8, weight_decay=0.0):

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError(f"Epsilon must be a non-negative float, but got {eps}")

        # Helper to convert various inputs to a uniform tensor format on the default device
        def _process_input(val, name):
            if isinstance(val, (float, int)):
                return val # Keep scalars as floats
            if isinstance(val, (list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        defaults = dict(lr=_process_input(lr, "learning rate"),
                        beta1=_process_input(beta1, "beta1"),
                        beta2=_process_input(beta2, "beta2"),
                        degree=_process_input(degree, 'degree'),
                        eps=eps,
                        weight_decay=_process_input(weight_decay, "weight_decay"))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with cross-network normalization."""
        for group in self.param_groups:
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_state = self.state[p]

                # State initialization and hyperparameter broadcasting
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    for name in ['lr', 'beta1', 'beta2', 'degree', 'weight_decay']:
                        val = group[name]
                        # If val is a tensor and param is batched, prepare for broadcasting
                        if isinstance(val, torch.Tensor) and p.ndim > 0 and val.numel() > 1:
                            param_state[f'{name}_prepared'] = val.to(p.device).view(-1, *[1] * (p.ndim - 1))
                        else:
                            param_state[f'{name}_prepared'] = val

                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                lr = param_state['lr_prepared']
                beta1 = param_state['beta1_prepared']
                beta2 = param_state['beta2_prepared']
                degree = param_state['degree_prepared']
                weight_decay = param_state['weight_decay_prepared']
                
                param_state['step'] += 1
                
                # Decoupled weight decay (AdamW style)
                if not isinstance(weight_decay, float) or weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # First moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Second moment estimate (v_t)
                grad_abs_p = torch.pow(grad.abs(), degree)
                exp_avg_sq.mul_(beta2).add_(grad_abs_p, alpha=1.0 - beta2)
                
                # Bias correction
                bias_correction1 = 1.0 - beta1 ** param_state['step']
                bias_correction2 = 1.0 - beta2 ** param_state['step']
                
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                
                # --- NORMALIZATION LOGIC ---

                # 1. Calculate the un-normalized update vector for the entire batch
                denom = torch.pow(v_hat, 1.0 / degree).add_(eps)
                unnormalized_update = m_hat / denom
                
                # Check if this parameter is batched and has more than one network
                is_batched = p.ndim > 0 and p.shape[0] > 1

                if is_batched:
                    # 2. Calculate the L2-norm of the update for each network in the batch
                    update_norms = torch.linalg.vector_norm(unnormalized_update, dim=tuple(range(1, unnormalized_update.ndim)))
                    
                    # 3. Get the reference norm (from the first network)
                    reference_norm = update_norms[0]
                    
                    # 4. Compute the scaling factor to match the reference norm
                    scale_factor = reference_norm / (update_norms + eps)
                    
                    # Reshape for broadcasting: from [n] to [n, 1, 1, ...]
                    scale_factor = scale_factor.view(-1, *[1] * (p.ndim - 1))
                    
                    # 5. Apply the scaled update
                    final_update = lr * unnormalized_update * scale_factor
                    p.add_(final_update, alpha=-1.0)
                else:
                    # Original update for non-batched or single-network parameters
                    final_update = lr * unnormalized_update
                    p.add_(final_update, alpha=-1.0)
                
class LazyAdamW(torch.optim.Optimizer):
    """
    Implements a from-scratch AdamW optimizer.
    
    Supports broadcasting of learning rate, betas, and weight decay
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    Epsilon (eps) is treated as a scalar float.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps: float = 1e-8, 
                 weight_decay=0.0):

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError(f"Epsilon must be a non-negative float, but got {eps}")

        def _process_input(val, name):
            # Helper to convert various inputs to a uniform tensor format
            if isinstance(val, (float, int, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate")
        processed_beta1 = _process_input(beta1, "beta1")
        processed_beta2 = _process_input(beta2, "beta2")
        processed_wd = _process_input(weight_decay, "weight_decay")

        defaults = dict(lr=processed_lr, beta1=processed_beta1, beta2=processed_beta2,
                        eps=eps, weight_decay=processed_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with cached checks."""
        for group in self.param_groups:
            # Eps is a scalar, retrieve it once per group
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_state = self.state[p]

                # one time move
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    # Cache only the hyperparameters that need broadcasting
                    for name in ['lr', 'beta1', 'beta2', 'weight_decay']:
                        val = group[name]
                        if isinstance(val, torch.Tensor):
                            param_state[f'{name}_prepared'] = val.to(p.device).view(-1, *[1] * (p.ndim - 1))

                        else:
                            param_state[f'{name}_prepared'] = val


                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                lr = param_state['lr_prepared']
                beta1 = param_state['beta1_prepared']
                beta2 = param_state['beta2_prepared']
                weight_decay = param_state['weight_decay_prepared']
                
                param_state['step'] += 1
                
                # find if there are gradients across all parameters and therefore update with momentum
                update_needed = (grad.sum(dim=tuple(range(1, p.ndim))) > 0).view(-1, *[1] * (p.ndim - 1))
                
                if not isinstance(weight_decay, float) or weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay * update_needed)
                
                # first moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad * (1 - beta1)) # alpha expects a scalar but beta1 might be a list

                # second moment estimate (v_t)
                exp_avg_sq.mul_(beta2).addcmul_(grad*grad, (1-beta2), value=1) # similarly, value expects a scalar
                

                # bias correction (\hat{m_t}, \hat{v_t})
                bias_correction1 = 1.0 - beta1 ** param_state['step']
                bias_correction2 = 1.0 - beta2 ** param_state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                step_size = lr / bias_correction1
                
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                
                corrected_exp_avg.mul_(step_size)
                
                # which networks actually have gradients and therefore require updating
                corrected_exp_avg.mul_(update_needed)
                p.addcdiv_(corrected_exp_avg, denom, value=-1) 
                
class LazySGD(torch.optim.Optimizer):
    """
    Implements a simple from-scratch SGD optimizer with optional momentum.
    
    This version supports broadcasting of learning rates and momentum values
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    """
    def __init__(self, params, lr, momentum=0.0):

        def _process_input(val, name):
            # Convert lists or numpy arrays to a torch tensor
            if isinstance(val, (float, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate")
        processed_momentum = _process_input(momentum, "momentum")

        defaults = dict(lr=processed_lr, momentum=processed_momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with cached checks."""
        for group in self.param_groups:
            # These are the original lr/momentum, which can be float or tensor
            base_lr = group['lr']
            base_momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                # ONE-TIME SETUP per parameter (lazy initialization)
                # This block runs only on the very first step for each parameter.
                if 'lr_prepared' not in param_state:
                    # 1. Prepare Learning Rate
                    if isinstance(base_lr, torch.Tensor):
                        # Move to device and reshape for broadcasting ONCE
                        prepared_lr = base_lr.to(p.device).view(-1, *[1] * (p.ndim - 1))  
                    else:
                        prepared_lr = base_lr
                    param_state['lr_prepared'] = prepared_lr

                    # 2. Prepare Momentum
                    if isinstance(base_momentum, torch.Tensor):
                        prepared_momentum = base_momentum.to(p.device).view(-1, *[1] * (p.ndim - 1))
                    else:
                        prepared_momentum = base_momentum
                    param_state['momentum_prepared'] = prepared_momentum
                    param_state['use_momentum'] = (isinstance(prepared_momentum, float) and prepared_momentum > 0.0) or \
                                   isinstance(prepared_momentum, torch.Tensor)

                # --- Use the cached, prepared values ---
                # On all subsequent steps, these lines retrieve the pre-processed values instantly.
                lr = param_state['lr_prepared']
                momentum = param_state['momentum_prepared']
                grad = p.grad
                
                # --- Momentum Update ---
                grad_update = grad
                use_momentum = param_state['use_momentum'] 
                
                if use_momentum:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.clone(grad).detach()
                    
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad_update = buf
                    
                # find if there are gradients across all parameters and therefore update with momentum
                update_needed = (grad.sum(dim=tuple(range(1, p.ndim))) > 0).view(-1, *[1] * (p.ndim - 1))
                grad_update.mul(update_needed) # zero-out those that shouldn't update

                p.addcmul_(grad_update, lr, value=-1)