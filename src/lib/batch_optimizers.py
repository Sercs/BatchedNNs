import torch
import numpy as np

class SGD(torch.optim.Optimizer):
    """
    Implements a simple from-scratch SGD optimizer with optional momentum.
    
    This version supports broadcasting of learning rates and momentum values
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    """
    def __init__(self, params, lr, momentum=0.0, device='cpu'):

        def _process_input(val, name, device):
            # Convert lists or numpy arrays to a torch tensor
            if isinstance(val, (float, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate", device)
        processed_momentum = _process_input(momentum, "momentum", device)

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
                                                        #                 |
                                                        # some lines of code feel like magical runes
                                                        # essentially [1,...,1] up to length p.ndim-1
                                                        # -1 skips network batch dim
                                                        # *[1] then "unpacks" this so we get 1,...,1
                                                        # thus .view(-1, 1, ..., 1)
                                                        # this accounts for ndim params 
                                                        # like bias (n, out) and weights (n, out, in)
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
                 weight_decay=0.0, device='cpu'):

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError(f"Epsilon must be a non-negative float, but got {eps}")

        def _process_input(val, name, device):
            # Helper to convert various inputs to a uniform tensor format
            if isinstance(val, (float, int, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate", device)
        processed_beta1 = _process_input(beta1, "beta1", device)
        processed_beta2 = _process_input(beta2, "beta2", device)
        processed_wd = _process_input(weight_decay, "weight_decay", device)

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
                        if isinstance(val, torch.Tensor) and val.shape:
                            param_state[f'{name}_prepared'] = val.to(p.device).view(-1, *[1] * (p.ndim - 1))
                                                                        #                 |
                                                                        # some lines of code feel like magical runes
                                                                        # essentially [1,...,1] up to length p.ndim-1
                                                                        # -1 skips network batch dim
                                                                        # *[1] then "unpacks" this so we get 1,...,1
                                                                        # thus .view(-1, 1, ..., 1)
                                                                        # this accounts for ndim params 
                                                                        # like bias (n, out) and weights (n, out, in)
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
                exp_avg_sq.mul_(beta2).addcmul_(grad*grad, (1-beta2), value=1) # similarly, value expects a scalar
                

                # bias correction (\hat{m_t}, \hat{v_t})
                bias_correction1 = 1.0 - beta1 ** param_state['step']
                bias_correction2 = 1.0 - beta2 ** param_state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                denom = (exp_avg_sq.div(bias_correction2)).sqrt_().add_(eps)
                
                step_size = lr / bias_correction1
                
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                
                corrected_exp_avg.mul_(step_size)
                p.addcdiv_(corrected_exp_avg, denom, value=-1) 
                
class LazyAdamW(torch.optim.Optimizer):
    """
    Implements a from-scratch AdamW optimizer.
    
    Supports broadcasting of learning rate, betas, and weight decay
    for parameters with a leading 'batch' dimension (e.g., shape [n, ...]).
    Epsilon (eps) is treated as a scalar float.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps: float = 1e-8, 
                 weight_decay=0.0, device='cpu'):

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError(f"Epsilon must be a non-negative float, but got {eps}")

        def _process_input(val, name, device):
            # Helper to convert various inputs to a uniform tensor format
            if isinstance(val, (float, int, list, np.ndarray)):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            if isinstance(val, torch.Tensor):
                if torch.any(val < 0.0):
                    raise ValueError(f"Invalid {name} value found in tensor: {val}")
                return val
            raise TypeError(f"{name} must be a float, list, np.ndarray, or torch.Tensor")

        processed_lr = _process_input(lr, "learning rate", device)
        processed_beta1 = _process_input(beta1, "beta1", device)
        processed_beta2 = _process_input(beta2, "beta2", device)
        processed_wd = _process_input(weight_decay, "weight_decay", device)

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
                        if isinstance(val, torch.Tensor) and val.shape:
                            param_state[f'{name}_prepared'] = val.to(p.device).view(-1, *[1] * (p.ndim - 1))
                                                                        #                 |
                                                                        # some lines of code feel like magical runes
                                                                        # essentially [1,...,1] up to length p.ndim-1
                                                                        # -1 skips network batch dim
                                                                        # *[1] then "unpacks" this so we get 1,...,1
                                                                        # thus .view(-1, 1, ..., 1)
                                                                        # this accounts for ndim params 
                                                                        # like bias (n, out) and weights (n, out, in)
                        else:
                            param_state[f'{name}_prepared'] = val


                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                lr = param_state['lr_prepared']
                beta1 = param_state['beta1_prepared']
                beta2 = param_state['beta2_prepared']
                weight_decay = param_state['weight_decay_prepared']
                
                param_state['step'] += 1
                
                
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
                
                denom = (exp_avg_sq.div(bias_correction2)).sqrt_().add_(eps)
                
                step_size = lr / bias_correction1
                
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                
                corrected_exp_avg.mul_(step_size)
                
                # which networks actually have gradients and therefore require updating
                corrected_exp_avg.mul_(update_needed)
                p.addcdiv_(corrected_exp_avg, denom, value=-1) 

