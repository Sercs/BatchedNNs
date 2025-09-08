# TODO: previous parameters as handlers, sleep epochs, weight masking

import torch
import numpy as np
import time
"""
    Observers represent any code you'd want to run during training but not
    necessarily during every training run you implement. They listen for 
    functions called in the main training loop to make modifications or 
    logging.
"""
class Observer:
    def __init__(self):
        # this will be populated by the Trainer upon initialization.
        self.trainer = None
    def get_x(self, state): pass # testing
    
    # main function listeners
    def before_train(self, state): pass
    def after_train(self, state): pass
    def before_epoch(self, state): pass
    def after_epoch(self, state): pass
    def before_test(self, state): pass
    def on_test_run(self, state): pass
    def after_test(self, state): pass
    def before_train_forward(self, state): pass
    def after_train_forward(self, state): pass
    def before_test_forward(self, state): pass
    def after_test_forward(self, state): pass
    def before_update(self, state): pass
    def after_update(self, state): pass
    def before_step(self, state): pass
    def after_step(self, state): pass
    
"""
    Handlers behave like Observers, listening for functions to do something.
    However, they only operate on each parameter as it comes. They are 
    typically paired with the ParamIterator(Observer) to make looping over 
    layers more efficient.
"""

class Handler:
    # >>> 
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

class Data(Observer):
    def before_train(self, state):
        state['data'] = {}
        
class TestLoop(Observer):
    """
    An observer that performs a test/validation loop over a given dataset.
    
    This observer listens for the 'on_test_run' event fired by the Trainer.
    """
    
    def __init__(self, name, test_dataloader, criterions, device='cpu'):
        self.name = name
        self.test_dataloader = test_dataloader
        self.criterions = {name: crit.to(device) for name, crit in criterions.items()}
        self.device = device
        self.trainer = None # This will be set by the Trainer

    def on_test_run(self, state):
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

                # since BatchLosses reduce by mean, we multiple by batch_size
                # and then we can average at the end
                # slightly hacky
                batch_losses = {name: crit(y_hat, y, idx)
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
        
class Timer(Observer):
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

class RunningLossTracker(Observer):
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

class TestingLossTracker(Observer):
    def __init__(self, test_names, criterion_names):
        super().__init__()
        if type(test_names) is not list:
            test_names = [test_names]

        if type(criterion_names) is not list:
            criterion_names = [criterion_names]
            
        self.criterion_names = criterion_names
        self.test_names = test_names

    def before_train(self, state):
        self.test_losses = {
            name: {crit: torch.zeros((state['n_networks'],)) for crit in self.criterion_names}
            for name in self.test_names
        }
        
        state['data']['test_losses'] = {
            name: {crit: [] for crit in self.criterion_names}
            for name in self.test_names
        }

    def after_test_forward(self, state):
        current_test_name = state['current_test_context']
        if current_test_name in self.test_names:
            losses = state[f'{current_test_name}_losses']
            for criterion_name in losses:
                loss_value = losses[criterion_name]
                self.test_losses[current_test_name][criterion_name] += loss_value.clone().detach().cpu()

    def after_test(self, state):
        for test_name in self.test_losses:
            for criterion_name in self.test_losses[test_name]:
                accumulated_loss = self.test_losses[test_name][criterion_name]
                state['data']['test_losses'][test_name][criterion_name].append(accumulated_loss.clone().numpy())
            self._reset_test(test_name, state)
            
    def _reset_test(self, test_name, state):
        self.test_losses[test_name] = {
            crit: torch.zeros((state['n_networks'],)) for crit in self.criterion_names
        }

class RunningAccuracyTracker(Observer):
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

class TestingAccuracyTracker(Observer):
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
        
class LossTracker(Observer):
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
        
class AccuracyTracker(Observer):
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
        
class ForwardPassCounter(Observer):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        n_networks = state['n_networks']
        self.forward_pass_counts = torch.zeros(n_networks, dtype=torch.long)
        state['data']['forward_pass_counts'] = []

    def after_update(self, state):
        idx = state.get('current_train_idx')
        padding_value = state.get('padding_value', -1)

        if idx is None:
            return

        is_valid = (idx != padding_value)
        passes_this_batch = is_valid.any(dim=0).long()
        
        self.forward_pass_counts += passes_this_batch.cpu()

    def after_test(self, state):
        state['data']['forward_pass_counts'].append(self.forward_pass_counts.clone().numpy())

class ForwardItemCounter(Observer):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        n_networks = state['n_networks']
        self.forward_item_counts = torch.zeros(n_networks, dtype=torch.long)
        state['data']['forward_item_counts'] = []

    def after_update(self, state):
        idx = state.get('current_train_idx')
        padding_value = state.get('padding_value', -1)

        if idx is None:
            return

        is_valid = (idx != padding_value)
        items_this_batch = is_valid.sum(dim=0).long()

        self.forward_item_counts += items_this_batch.cpu()

    def after_test(self, state):
        """Logs the current counts to the data dictionary."""
        state['data']['forward_item_counts'].append(self.forward_item_counts.clone().numpy())
        
class BackwardPassCounter(Observer):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        self.backward_pass_counts = torch.zeros(state['n_networks'], dtype=torch.long)
        state['data']['backward_pass_counts'] = []

    def after_update(self, state):
        per_sample_loss = state.get('per_sample_losses')
        if per_sample_loss is None: return
        
        was_updated_mask = (per_sample_loss > 1e-9) # eps for rounding errors
        
        updates_this_batch = was_updated_mask.any(dim=0).long()
        self.backward_pass_counts += updates_this_batch.cpu()

    def after_test(self, state):
        state['data']['backward_pass_counts'].append(self.backward_pass_counts.clone().numpy())

class BackwardItemCounter(Observer):
    def __init__(self):
        super().__init__()

    def before_train(self, state):
        self.backward_item_counts = torch.zeros(state['n_networks'], dtype=torch.long)
        state['data']['backward_item_counts'] = []

    def after_update(self, state):
        per_sample_loss = state.get('per_sample_losses')
        if per_sample_loss is None: return

        was_updated_mask = (per_sample_loss > 1e-9) # eps for rounding errors

        updated_items_this_batch = was_updated_mask.sum(dim=0).long()
        self.backward_item_counts += updated_items_this_batch.cpu()

    def after_test(self, state):
        state['data']['backward_item_counts'].append(self.backward_item_counts.clone().numpy())
        
class PerSampleBackwardCounter(Observer):
    def __init__(self, max_samples):
        super().__init__()
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")
        self.max_samples = max_samples

    def before_train(self, state):
        n_networks = state['n_networks']
        self.per_sample_backward_counts = torch.zeros((self.max_samples, n_networks), dtype=torch.long)
        state['data']['per_sample_backward_counts'] = self.per_sample_backward_counts

    def after_update(self, state):
        per_sample_loss = state.get('per_sample_losses').cpu()
        idx = state.get('current_train_idx').cpu()

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
        
class PerNetworkLearningRate(Observer):
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
    
class L1Regularizer(Observer):
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

class L2Regularizer(Observer):
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
        
 # stops repeated memory (i.e. if multiple observers need previous parameters)
class PreviousParameterProvider(Observer):
    def __init__(self):
        super().__init__() #                                   need model for intialization 
        self.previous_parameters = {}
        
    def before_update(self, state): #                          now we need the state
        self.previous_parameters = {n: p.clone().detach() for n, p in state['model'].named_parameters()}
    
    def after_update(self, state):
        self.previous_parameters = {} # save GPU memory
        
    def get_parameters(self):
        return self.previous_parameters
    
class PreviousPreviousParameterProvider(Observer):
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
class InitialParameterProvider(Observer):
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
   
class MinimumEnergyL1NetworkTracker(Observer):
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train(self, state):
        self.minimum_energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l1'] = []
        
    def after_step(self, state):       
        delta = torch.zeros((state['n_networks'])).to(state['device'])
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n] 
            delta += (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))
        self.minimum_energy_l1 = delta.detach().cpu()
        
    def after_test(self, state):
        state['data']['minimum_energies_l1'].append(self.minimum_energy_l1.clone().numpy())
 
class EnergyL1NetworkTracker(Observer):
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
        
class MinimumEnergyL1LayerwiseTracker(Observer):
    def __init__(self, initial_parameter_provider):
        super().__init__()
        self.provider = initial_parameter_provider
    
    def before_train(self, state):

        self.minimum_energy_l1_layerwise = {
            n: torch.zeros((state['n_networks'],)) for n, _ in state['model'].named_parameters()
        }
        state['data']['minimum_energies_l1_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
        
    def after_step(self, state):
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n].to(p.device)
            delta = (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))

            self.minimum_energy_l1_layerwise[n] = delta.detach().cpu()
        
    def after_test(self, state):
        for n, energy in self.minimum_energy_l1_layerwise.items():
            state['data']['minimum_energies_l1_layerwise'][n].append(energy.clone().numpy())

class EnergyL1LayerwiseTracker(Observer):
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
            
class MinimumEnergyL1NeuronwiseTracker(Observer):
    def __init__(self, initial_parameter_provider, energy_direction='incoming'):
        super().__init__()
        self.provider = initial_parameter_provider
        if isinstance(energy_direction, str):
            energy_direction = [energy_direction]
        self.energy_direction = energy_direction
        
    def before_train(self, state):
        self.minimum_energy_l1_neuronwise = {}
        state['data']['minimum_energies_l1_neuronwise'] = {}
        for n, p in state['model'].named_parameters():
            state['data']['minimum_energies_l1_neuronwise'][n] = {
                e_dir: [] for e_dir in self.energy_direction
            }
            
    def after_step(self, state):
        for n, p in state['model'].named_parameters():
            self.minimum_energy_l1_neuronwise[n] = {}
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
                    self.minimum_energy_l1_neuronwise[n][e_dir] = energy.detach().cpu()
        
    def after_test(self, state):
        for n, layer_energies in self.minimum_energy_l1_neuronwise.items():
            for e_dir, energy in layer_energies.items():
                state['data']['minimum_energies_l1_neuronwise'][n][e_dir].append(energy.clone().numpy())

class EnergyL1NeuronwiseTracker(Observer):

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
        
class MinimumEnergyL2NetworkTracker(Observer):
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train(self, state):
        self.minimum_energy_l2 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l2'] = []
        
    def after_step(self, state):       
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
        
    def after_test(self, state):
        state['data']['minimum_energies_l2'].append(self.minimum_energy_l2.clone().numpy())   
        
class EnergyL2NetworkTracker(Observer):
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
        
class EnergyL0NetworkTracker(Observer):
    def before_train(self, state):
        self.energy_l0 = torch.zeros((state['n_networks'],))
        state['data']['energies_l0'] = []
        
        # before step, after loss = grads available
    def before_step(self, state):
        for n, p in state['model'].named_parameters():
            delta = torch.count_nonzero(p.grad, dim=(tuple(range(1, p.ndim))))
            self.energy_l0 += delta.detach().cpu()
    
    def after_step(self, state):
        state['data']['energies_l0'].append(self.energy_l0.clone().numpy())
        
class MinimumEnergyL0NetworkTracker(Observer):
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
        
    def before_train(self, state):
        self.minimum_energy_l0 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l0'] = []
        
        # before step, after loss = grads available
    def before_step(self, state):
        delta = torch.zeros((state['n_networks'],)).to(state['device'])
        for n, p in state['model'].named_parameters():
            init_p = self.provider.initial_parameters[n] 
            delta += torch.count_nonzero(p - init_p, dim=(tuple(range(1, p.ndim))))
        self.minimum_energy_l0 = delta.detach().cpu()
    
    def after_step(self, state):
        state['data']['minimum_energies_l0'].append(self.minimum_energy_l0.clone().numpy())
        

"""
    ParameterIterator is designed to make Observers that require looping over
    parameters more flexible and more efficient. We might make an
    Observer for each data we're interested in. However, if these loop over
    parameters then we make multiple calls for a loop. Instead, ParameterIterator
    takes multiple of these Observers (termed Handlers) and calls them for each
    parameter.
"""
class ParameterIterator(Observer):
    def __init__(self, handlers):
        self.handlers = handlers
        
    def _func(self, state, event_name):
        event_name += '_func'
        #print(event_name)
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
        
    def after_test(self, state):
        if self._should_run('func', 'after_test'):
            self._func(state, 'after_test')
        
        if self._should_run('log', 'after_test'):
            self._log(state, 'after_test')
            
class EnergyL0NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'before_step' : ['func'],
            'after_test' : ['log']
        }

    def before_train_log(self, state):
        self.energy_l0 = torch.zeros((state['n_networks'],))
        state['data']['energies_l0'] = []
        
        # before step, after loss = grads available
    def before_step_func(self, n, p, state):
        delta = torch.count_nonzero(p.grad, dim=(tuple(range(1, p.ndim))))
        self.energy_l0 += delta.detach().cpu()
    
    def after_test_log(self, state):
        state['data']['energies_l0'].append(self.energy_l0.clone().numpy())
        
class MinimumEnergyL0NetworkHandler(Handler):
    
        
    events = {
            'before_train' : ['log'],
            'before_update' : ['log'],
            'before_step' : ['func'],
            'after_update' : ['log'],
            'after_test' : ['log']
        }

    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
        
    def before_train_log(self, state):
        self.minimum_energy_l0 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l0'] = []
        
    def before_update_log(self, state):
        self.delta = torch.zeros((state['n_networks'],)).to(state['device'])
        
        # before step, after loss = grads available
    def before_step_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n] 
        self.delta += torch.count_nonzero(p - init_p, dim=(tuple(range(1, p.ndim))))
        
    def after_update_log(self, state):
        self.minimum_energy_l0 = self.delta.detach().cpu()
    
    def after_test_log(self, state):
        state['data']['minimum_energies_l0'].append(self.minimum_energy_l0.clone().numpy())

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
        #print(delta)
        self.energy_l1 += delta.detach().cpu()
        
    def after_test_log(self, state):
        state['data']['energies_l1'].append(self.energy_l1.clone().numpy())
        
class MinimumEnergyL1NetworkHandler(Handler):
    
    events = {
            'before_train' : ['log'],
            'before_update' : ['log'],
            'after_step' : ['func'],
            'after_update' : ['log'],
            'after_test' : ['log']
        }
    
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train_log(self, state):
        self.minimum_energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l1'] = []
    
    def before_update_log(self, state):
        self.delta = torch.zeros((state['n_networks'],)).to(state['device'])
    
    def after_step_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n] 
        self.delta += (p - init_p).abs().sum(dim=tuple(range(1, p.ndim)))
        
    def after_update_log(self, state):
        self.minimum_energy_l1 = self.delta.detach().cpu()
        
    def after_test_log(self, state):
        state['data']['minimum_energies_l1'].append(self.minimum_energy_l1.clone().numpy())

class MinimumEnergyL1LayerwiseHandler(Handler):
    events = {
        'before_train': ['log'],
        'before_step': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
    }
    
    def __init__(self, initial_parameter_provider):
        super().__init__()
        self.provider = initial_parameter_provider

    def before_train_log(self, state):
        state['data']['minimum_energies_l1_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
    
    def before_step_log(self, state):
        self.current_minimum_energy = {}

    def after_step_func(self, n, p, state):
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
        'before_step': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
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
            
    def before_step_log(self, state):
        self.current_minimum_energy = {}

    def after_step_func(self, n, p, state):
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
            'before_update' : ['log'],
            'after_step' : ['func'],
            'after_update' : ['log'],
            'after_test' : ['log']
        }
    
    def __init__(self, initial_parameter_provider):
        self.provider = initial_parameter_provider
    
    def before_train_log(self, state):
        self.minimum_energy_l1 = torch.zeros((state['n_networks'],))
        state['data']['minimum_energies_l2'] = []
    
    def before_update_log(self, state):
        self.delta = torch.zeros((state['n_networks'],)).to(state['device'])
    
    def after_step_func(self, n, p, state):
        init_p = self.provider.initial_parameters[n] 
        self.delta += ((
                        (p - init_p)**2
                             ).sum(dim=tuple(range(1, p.ndim))
                                 )
                        )
        
    def after_update_log(self, state):
        self.delta = self.delta ** 0.5
        self.minimum_energy_l2 = self.delta.detach().cpu()
        
    def after_test_log(self, state):
        state['data']['minimum_energies_l2'].append(self.minimum_energy_l2.clone().numpy())
        
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
        'before_step': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
    }
    
    def __init__(self, initial_parameter_provider):
        super().__init__()
        self.provider = initial_parameter_provider

    def before_train_log(self, state):
        state['data']['minimum_energies_l2_layerwise'] = {
            n: [] for n, _ in state['model'].named_parameters()
        }
    
    def before_step_log(self, state):
        self.current_minimum_energy = {}

    def after_step_func(self, n, p, state):
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
        'before_step': ['log'],
        'after_step': ['func'],
        'after_test': ['log']
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
            
    def before_step_log(self, state):
        self.current_minimum_energy = {}

    def after_step_func(self, n, p, state):
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