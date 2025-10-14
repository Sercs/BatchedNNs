import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# TODO: I think MSE should always average the outputs and reduction only applies to batch
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
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
        self.reduction = reduction # how to reduce last dim
        # initialize the core loss function to get per-element losses
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, idx=None, padding_value=-1):
        unreduced_loss = self.loss_fn(y_hat, y)


        mask = (idx != padding_value).float()  # get padded items
        if self.reduction == 'mean':
            loss = torch.mean(unreduced_loss, dim=-1) * mask
        else: # self.reduction == 'sum'
            loss = torch.sum(unreduced_loss, dim=-1) * mask

        return loss
    
class MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
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
        # initialize the core loss function to get per-element losses
        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, y_hat, y, idx=None, padding_value=-1):
        unreduced_loss = self.loss_fn(y_hat, y)
            
        mask = (idx != padding_value).float()  # get padded items
        if self.reduction == 'mean':
            loss = torch.mean(unreduced_loss, dim=-1) * mask
        else: # self.reduction == 'sum'
            loss = torch.sum(unreduced_loss, dim=-1) * mask

        return loss

class HingeLoss(nn.Module): # conventionally Hinge is reduced by sum
    def __init__(self, reduction='sum', func=nn.ReLU(), margin=1.0):
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
        self.func = func
        
        if isinstance(margin, torch.Tensor) and len(margin.shape) == 0:
            self.margin = margin.unsqueeze(0).unsqueeze(-1)
        else:
            self.margin = torch.tensor(margin).unsqueeze(0).unsqueeze(-1)

    def forward(self, y_hat, y, idx=None, padding_value=-1):
        self.margin = self.margin.to(y_hat.device)
        target_activities = torch.sum(y_hat * y, dim=-1, keepdim=True)
        margins = self.margin + y_hat - target_activities
        loss_terms = self.func(margins)
        unreduced_loss = loss_terms * (y < 1)
            
        mask = (idx != padding_value).float()  # get padded items
        if self.reduction == 'mean':
            loss = torch.mean(unreduced_loss, dim=-1) * mask
        else: # self.reduction == 'sum'
            loss = torch.sum(unreduced_loss, dim=-1) * mask
        return loss

class CrossEntropyLoss(nn.Module): # cross-entropy already reduces last dim 
                                   # this is only here for consistency
   #                                 |                                
    def __init__(self, reduction='mean', confidence_threshold=0.0): 
                                          
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
        # initialize the core loss function to get per-element losses
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.confidence_threshold = confidence_threshold

    def forward(self, y_hat, y, idx=None, padding_value=-1):
        unreduced_loss = self.loss_fn(y_hat.transpose(1, -1), y.transpose(1, -1))
        if self.confidence_threshold > 0:
            probs = F.softmax(y_hat, dim=-1)
            confidence = (probs * y).sum(-1)
            confidence_mask = (confidence < self.confidence_threshold).float()
            unreduced_loss = unreduced_loss * confidence_mask
        mask = (idx != padding_value).float() # get padded items

        loss = unreduced_loss * mask # cross entropy is already per-sample
        return loss

class LazyLoss(nn.Module):
    def __init__(self, loss_fn, reduction=None): # the wrapped loss function should handle reduction
        super().__init__()                         # only here for consistency
        self.wrapped_loss_fn = loss_fn 
        self.reduction = reduction
        
    def forward(self, y_hat, y, idx=None, padding_value=-1):
        update = (y_hat.argmax(-1) != y.argmax(-1))
        per_sample_losses = self.wrapped_loss_fn(y_hat, y, idx) * update
        
        mask = (idx != padding_value).float()  # get padded items

        loss = per_sample_losses * mask
        return loss
    
class StatefulLazyLoss(nn.Module):     # like lazy, the wrapped loss function should handle reduction
    def __init__(self, loss_fn, max_samples, n_networks, reduction='mean'):
        super().__init__()                         
        self.wrapped_loss_fn = loss_fn
        self.memory = torch.zeros((max_samples, n_networks), dtype=torch.long)
        self.model_idxs = torch.arange(0, n_networks, dtype=torch.long)
        
        self.reduction = reduction
        
    def forward(self, y_hat, y, idx=None, padding_value=-1):
        
        self.memory = self.memory.to(y_hat.device)
        self.model_idxs = self.model_idxs.to(y_hat.device)
        

        incorrect = torch.where(
            (y_hat.argmax(-1) != y.argmax(-1)) & (idx != padding_value)
        )

        self.memory.index_put_( # <- Gemini-Pro 2.5 tip
            (idx[incorrect], incorrect[1]),
            torch.tensor(1, device=y.device),
            accumulate=True
        )
        
        update = self.memory[idx, self.model_idxs] > 0
        per_sample_losses = self.wrapped_loss_fn(y_hat, y, idx) * update
        
        mask = (idx != padding_value).float()  # get padded items

        loss = per_sample_losses * mask

        return loss
        
        