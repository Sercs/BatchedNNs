import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# TODO: I think MSE should always average the outputs and reduction only applies to batch
class MSELoss(nn.Module):
    def __init__(self, per_sample=False, reduction='mean'):
        """
        Args:
            reduction (str): The reduction operation to apply: 'mean' or 'sum'.
            per_sample (bool): If True, reduces over feature dimensions to get a loss
                               per sample. If False, reduces over all dimensions to
                               get a single loss for the batch.
        """
        super().__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean' or 'sum'.")
        self.reduction = reduction
        self.per_sample = per_sample
        # initialize the core loss function to get per-element losses
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, idx=None):
        unreduced_loss = self.loss_fn(y_hat, y)
        if self.per_sample:
            # reduce only output dim
            dims_to_reduce = -1
        else:
            # reduce batch and output dim
            dims_to_reduce = (0, -1)


        if self.reduction == 'mean':
            loss = torch.mean(unreduced_loss, dim=dims_to_reduce)
        else: # self.reduction == 'sum'
            loss = torch.sum(unreduced_loss, dim=dims_to_reduce)

        return loss
    
class MAELoss(nn.Module):
    def __init__(self, per_sample=False, reduction='mean'):
        """
        Args:
            reduction (str): The reduction operation to apply: 'mean' or 'sum'.
            per_sample (bool): If True, reduces over feature dimensions to get a loss
                               per sample. If False, reduces over all dimensions to
                               get a single loss for the batch.
        """
        super().__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean' or 'sum'.")
        self.reduction = reduction
        self.per_sample = per_sample
        # initialize the core loss function to get per-element losses
        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, y_hat, y, idx=None):
        unreduced_loss = self.loss_fn(y_hat, y)
        if self.per_sample:
            # reduce only output dim
            dims_to_reduce = -1
        else:
            # reduce batch and output dim
            dims_to_reduce = (0, -1)


        if self.reduction == 'mean':
            loss = torch.mean(unreduced_loss, dim=dims_to_reduce)
        else: # self.reduction == 'sum'
            loss = torch.sum(unreduced_loss, dim=dims_to_reduce)

        return loss

class HingeLoss(nn.Module):
    def __init__(self, per_sample=False, reduction='sum', func=nn.ReLU(), margin=1.0):
        """
        Args:
            reduction (str): The reduction operation to apply: 'mean' or 'sum'.
            per_sample (bool): If True, reduces over feature dimensions to get a loss
                               per sample. If False, reduces over all dimensions to
                               get a single loss for the batch.
            margin (float or list): the safety margin required between the target neuron 
                            activity and incorrect neuron activities.
        """
        super().__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean' or 'sum'.")
        self.reduction = reduction
        self.per_sample = per_sample
        self.func = func
        
        if isinstance(margin, torch.Tensor) and len(margin.shape) == 1:
            margin = margin.unsqueeze(0).unsqueeze(-1)
        elif isinstance(margin, (np.ndarray, list)):
            #                             batch_dim    output_dim 
            margin = torch.tensor(margin).unsqueeze(0).unsqueeze(-1)
        self.margin = margin

    def forward(self, y_hat, y, idx=None):
        self.margin = self.margin.to(y_hat.device)
        target_activities = torch.sum(y_hat * y, dim=-1, keepdim=True)
        margins = self.margin + y_hat - target_activities
        loss_terms = self.func(margins)
        unreduced_loss = loss_terms * (y < 1)
        if self.per_sample:
            dims_to_reduce = -1
        else:
            dims_to_reduce = (0, -1)
        if self.reduction == 'mean':
            loss = torch.mean(unreduced_loss, dim=dims_to_reduce)
        else: # self.reduction == 'sum'
            loss = torch.sum(unreduced_loss, dim=dims_to_reduce)
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, per_sample=False, reduction='mean'):
        """
        Args:
            reduction (str): How to reduce the batch (sum them or mean them)
            per_sample (bool): If True, loss computed per sample. If False, reduces 
            over batch dim to get a single loss for the batch. Useful for 
            testing loops that expect losses over network dim.
        """
        super().__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean' or 'sum'.")
        self.reduction = reduction
        self.per_sample = per_sample
        # initialize the core loss function to get per-element losses
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_hat, y, idx=None):
        unreduced_loss = self.loss_fn(y_hat.transpose(1, -1), y.transpose(1, -1))
        if self.per_sample:
            loss = unreduced_loss # cross entropy is already per-sample
        else:
            if self.reduction == 'mean':
                loss = torch.mean(unreduced_loss, dim=0)
            else:
                loss = torch.sum(unreduced_loss, dim=0)
        return loss

class LazyLoss(nn.Module):
    def __init__(self, loss_fn, reduction='mean', per_sample=True, padding_value=-1):
        super().__init__()
        self.wrapped_loss_fn = loss_fn
        self.padding_value = padding_value
        if not self.wrapped_loss_fn.per_sample:
            raise Exception("Lazy methods require the wrapped loss to return sample-wise losses")
            
        self.per_sample = per_sample
        self.reduction = reduction
        
    def forward(self, y_hat, y, idx=None):
        update = (y_hat.argmax(-1) != y.argmax(-1))
        per_sample_losses = self.wrapped_loss_fn(y_hat, y, idx) * update
        if self.per_sample:
            loss = per_sample_losses
        else:
            if self.reduction == 'mean':
                loss = torch.mean(per_sample_losses, dim=0)
            else:
                loss = torch.sum(per_sample_losses, dim=0)
        return loss
    
class StatefulLazyLoss(nn.Module):
    def __init__(self, loss_fn, max_samples, n_networks, reduction='mean', per_sample=True, padding_value=-1):
        super().__init__()
        self.wrapped_loss_fn = loss_fn
        self.padding_value = padding_value
        self.memory = torch.zeros((max_samples, n_networks), dtype=torch.long)
        self.model_idxs = torch.arange(0, n_networks, dtype=torch.long)
        
        self.per_sample = per_sample
        self.reduction = reduction
        
        if not self.wrapped_loss_fn.per_sample:
            raise Exception("Lazy methods require the wrapped loss to return sample-wise losses")
        
    def forward(self, y_hat, y, idx=None):
        
        self.memory = self.memory.to(y_hat.device)
        self.model_idxs = self.model_idxs.to(y_hat.device)
        

        incorrect = torch.where(
            (y_hat.argmax(-1) != y.argmax(-1)) & (idx != self.padding_value)
        )
        #    Gemini-Pro 2.5 tip
        #             |
        #             V
        self.memory.index_put_(
            (idx[incorrect], incorrect[1]),
            torch.tensor(1, device=y.device),
            accumulate=True
        )
        
        update = self.memory[idx, self.model_idxs] > 0
        per_sample_losses = self.wrapped_loss_fn(y_hat, y, idx) * update
        
        if self.per_sample:
            loss = per_sample_losses
        else:
            if self.reduction == 'mean':
                loss = torch.mean(per_sample_losses, dim=0)
            else:
                loss = torch.sum(per_sample_losses, dim=0)
        return loss
        
        