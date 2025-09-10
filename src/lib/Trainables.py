# batch linear
# models
import torch
import torch.nn as nn

import numpy as np
import math
 
class BatchLinear(nn.Module):
    def __init__(self, n_linears, n_in, n_out, activation=nn.Identity(), add_residual=False, n_recurs=0, init_method=None, init_config={}):
        super().__init__()
        self.n_linears = n_linears
        self.n_in, self.n_out = n_in, n_out
        self.activation = activation
        
        if add_residual:
            assert n_in == n_out, 'Expected input dims == output dims when using residuals'
        self.add_residual = add_residual
        
        if n_recurs > 0:
            assert n_in == n_out, 'Expected input dims == output dims when using recurrents'
        self.n_recurs = n_recurs
        
        self.weights = nn.Parameter(torch.empty((n_linears, n_out, n_in)))
        self.biases = nn.Parameter(torch.zeros((n_linears, n_out)))
        
        _init_weights(self.weights, init_method, **init_config)
                
    def forward(self, x):
        if self.add_residual:
            for _ in range(self.n_recurs+1):
                x = x + self.activation(torch.einsum('bni,nji->bnj', x, self.weights) + self.biases)
        else:
            for _ in range(self.n_recurs+1):
                x = self.activation(torch.einsum('bni,nji->bnj', x, self.weights) + self.biases)
        return x

# TODO: it may be possible to make this an Interceptor           
class BatchLinearMasked(nn.Module):
    def __init__(self, n_linears, n_ins, n_outs, activation=nn.Identity(), add_residual=False, n_recurs=0, init_method=None, init_config={}):
        super().__init__()
        
        self.n_linears = n_linears
        if type(n_ins) is not list:
            n_ins = [n_ins]*n_linears
        if type(n_outs) is not list:
            n_outs = [n_outs]*n_linears
        n_in, n_out = max(n_ins), max(n_outs)
        self.n_in, self.n_out = n_in, n_out
        self.n_ins, self.n_outs = n_ins, n_outs
        self.activation = activation
        
        if add_residual:
            assert n_in == n_out, 'Expected input dims == output dims when using residuals'
        self.add_residual = add_residual
        
        if n_recurs > 0:
            assert n_in == n_out, 'Expected input dims == output dims when using recurrents'
        self.n_recurs = n_recurs
        
        self.weights = nn.Parameter(torch.zeros((n_linears, n_out, n_in)))
        self.biases = nn.Parameter(torch.zeros((n_linears, n_out)))
        
        self._init_weights(init_method, **init_config)
        self._init_masks()     
        
    def register_gradient_hooks(self):
        self.weights.register_hook(lambda grad: grad * self.weight_mask)
        self.biases.register_hook(lambda grad: grad * self.bias_mask)
        
    def _init_masks(self):
        w_mask = torch.zeros((self.n_linears, self.n_out, self.n_in))
        b_mask = torch.zeros((self.n_linears, self.n_out))
        with torch.no_grad():
            for i, w in enumerate(self.weights):
                # mask incoming
                n_active_in, n_active_out = self.n_ins[i], self.n_outs[i]
                w_mask[i][:n_active_out, :n_active_in] = torch.ones((n_active_out, n_active_in))
                b_mask[i][:n_active_out] = torch.ones((n_active_out,))
        self.register_buffer('weight_mask', w_mask)
        self.register_buffer('bias_mask', b_mask)
    
    # since we zero out parts of the tensor we need special handling for fan_in/fan_out.
    # TODO: update with list kwargs.
    def _init_weights(self, method, **kwargs):
        if method is None:
            method = 'kaiming_uniform'
            kwargs.setdefault('a', math.sqrt(5))
            
        init_map = {
            'uniform': nn.init.uniform_,
            'normal': nn.init.normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'zeros': nn.init.zeros_,
            'ones': nn.init.ones_,
        }

        method = method.lower()
        
        with torch.no_grad():
            n_linears = len(self.weights)
            for i, w in enumerate(self.weights):
                current_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, (list, np.ndarray)):
                        if len(value) != n_linears:
                            raise ValueError(
                                f"Length of kwarg '{key}' ({len(value)}) must match "
                                f"the number of weights ({n_linears})."
                            )
                        current_kwargs[key] = value[i]
                    else:
                        current_kwargs[key] = value
                n_active_in, n_active_out = self.n_ins[i], self.n_outs[i]
                init_map[method](w[:n_active_out, :n_active_in], **current_kwargs)
            
        # method = method.lower()
        # with torch.no_grad():
        #     for i, w in enumerate(self.weights):
        #         # mask incoming
        #         n_active_in, n_active_out = self.n_ins[i], self.n_outs[i]
        #         init_map[method](w[:n_active_out, :n_active_in], **kwargs)
                
    def forward(self, x):
        if self.add_residual:
            for _ in range(self.n_recurs+1):
                x = x + self.activation(torch.einsum('bni,nji->bnj', x, self.weights * self.weight_mask) 
                                        + self.biases * self.bias_mask)
        else:
            for _ in range(self.n_recurs+1):
                x = self.activation(
                    torch.einsum('bni,nji->bnj', x, self.weights * self.weight_mask) 
                                    + self.biases * self.bias_mask)
        return x


# Gemini-Pro used for handling list-like kwargs
def _init_weights(weights, method, **kwargs):
    if method is None:
        method = 'kaiming_uniform'
        kwargs.setdefault('a', math.sqrt(5))

    init_map = {
        'uniform': nn.init.uniform_,
        'normal': nn.init.normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'zeros': nn.init.zeros_,
        'ones': nn.init.ones_,
    }

    # remove kwargs that nn.init functions don't expect
    clone_weights = kwargs.pop('clone', False)
    scale = kwargs.pop('scale', None)

    method = method.lower()
    
    with torch.no_grad():
        if clone_weights:
            # Initialize the first tensor. This will serve as our "base" weights.
            init_map[method](weights[0], **kwargs)

            # First, copy the initialized base weights to all other tensors.
            for i in range(1, len(weights)):
                weights[i].copy_(weights[0])
            
            # If a 'scale' list is provided, apply it to the cloned weights.
            if scale is not None:
                if not isinstance(scale, (list, np.ndarray)) or len(scale) != len(weights):
                    raise ValueError(
                        f"The 'scale' kwarg must be a list or array with the "
                        f"same length as the number of weights ({len(weights)})."
                    )
                # Now, multiply each tensor by its corresponding scale factor.
                for i, w in enumerate(weights):
                    w.mul_(scale[i])

        else:
            n_linears = len(weights)
            for i, w in enumerate(weights):
                current_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, (list, np.ndarray)):
                        if len(value) != n_linears:
                            raise ValueError(
                                f"Length of kwarg '{key}' ({len(value)}) must match "
                                f"the number of weights ({n_linears})."
                            )
                        current_kwargs[key] = value[i]
                    else:
                        current_kwargs[key] = value
                
                init_map[method](w, **current_kwargs)
        