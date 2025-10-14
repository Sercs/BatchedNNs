# train step
# train loop
# test loop
import torch
import numpy as np
import json

def _json_converter(o):
    if isinstance(o, (torch.Tensor, np.ndarray)):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# TODO: reduction on trainer
class Trainer():
    def __init__(self, model, n_networks, optimizer, criterion, train_dataloader, test_dataloader, trackers=None, padding_value=-1, device='cpu'):
        self.model = model
        self.n_networks = n_networks
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.padding_value = padding_value
        if trackers is None:
            self.trackers = []
        else:
            self.trackers = trackers
            for t in trackers:
                t.trainer = self
        self.state = {
                'model' : self.model,
                'optimizer' : self.optimizer,
                'criterion' : self.criterion,
                'n_networks' : n_networks, # can't think of a better method for this
                'step' : 1,
                'device' : device,
                'padding_value': padding_value,
                'data' : {}
            }

        self.device = device

                                                     #    event name the observer looks out for (i.e. on_epoch_end)
    def _fire_event(self, event_name):               #                         |
        for tracker in self.trackers:                #                         V
            getattr(tracker, event_name)(self.state) # effectively observer.event_name(self.state)
    
    def train_loop(self, n_epochs, test_interval, dataset_size=None, sample_increment=None):
        
        # >>> Initialization <<<
        if dataset_size is None:
            n_samples = len(self.train_dataloader.dataset)
        else:
            n_samples = dataset_size

        n_batches = len(self.train_dataloader)
        
        if type(test_interval) is float:
            test_interval = int(test_interval * n_samples)
        # computes stopping mid-epoch
        # good for quick tests (i.e. n_epochs = 0.01) 
        if type(n_epochs) is float:
            stop_on_sample = int(n_epochs * n_samples)
        else:
            stop_on_sample = n_epochs # n_epochs is whole number, therefore no need to stop mid-epoch

        test_epoch = 0
        sample_counter = 0

        # >>> Main loop <<<
        self._fire_event('before_train')
        for epoch in range(int(np.ceil((n_epochs)))):
            print(f"Epoch: {epoch+1}")
            self.state['epoch'] = epoch+1
            self._fire_event('before_epoch')
            for (x, y, idx) in self.train_dataloader:
                batch_size = x.size(0) # handles varying batch_size
                if sample_increment is None:
                    test_step = batch_size
                else:
                    test_step = sample_increment
                # if network A views 1 sample and network B views 32 samples
                # who should we use as a reference for testing?
                # if we test 5% of an epoch, network B operates 32x faster.
                
                # to handle varying batch_size over network dim
                # we do so with a manual override that indicates the reference
                # speed for testing (we could use min(batch) or mean(batch)).
                    
                sample_counter += test_step
                test_epoch += test_step
                self._fire_event('before_update') # I like to use this for logging initializations
                step_loss, step_accuracy = self.train_step(x, y, idx)
                self.state['running_loss'] = step_loss.detach().cpu() # multiply by batch size since we average
                self.state['running_accuracy'] = step_accuracy.detach().cpu()
                self._fire_event('after_update')
                if test_epoch >= test_interval: # TODO: it may be possible to avoid this check and instead 
                                                #       have interceptors that need it look for a state['count']
                    print(f"[{sample_counter / stop_on_sample*100}%] > Forward passes: {self.state['step']} | Samples seen: {sample_counter}")            
                    self._fire_event('before_test') # initialization stuff usually
                    self._fire_event('on_test_run') # used primarily by the test loop 
                    self._fire_event('after_test')  # typically for recording metrics
                    test_epoch = 0
                if n_epochs % 1.0 != 0 and sample_counter > stop_on_sample:
                    self._fire_event('after_train') # mid epoch stop
                    return
                self.state['step'] += 1
            self._fire_event('after_epoch') # record stuff
        self._fire_event('after_train') # record stuff

    def train_step(self, x, y, idx):
        x, y, idx = x.to(self.device), y.to(self.device), idx.to(self.device) 
        if len(x.shape) < 3:
            x, y = x.unsqueeze(1), y.unsqueeze(1).repeat((1, self.n_networks, 1))
            idx = idx.unsqueeze(1).repeat(1, self.n_networks)
            
        # note that these will be device bound
        self.state['x'], self.state['y'], self.state['idx'] = x, y, idx # useful for data augmentation
        
        self._fire_event('before_train_forward') # do stuff like data augmentation or idx tracking here
        y_hat = self.model(x)
        
        # if we get loss here, we can modify it with observers
        self.optimizer.zero_grad()
        
        per_sample_supervised_loss = self.criterion(y_hat, y, idx, self.state['padding_value'])
        self.state['per_sample_losses'] = per_sample_supervised_loss # useful for computing which samples were used
        # this code essentially averages per batch accounting for buffered items
        mask = (idx != self.padding_value) # get padded items
        #masked_losses = per_sample_supervised_loss * mask # mask them
        n_valid_samples = mask.sum(0) # get the total items 
        supervised_loss = per_sample_supervised_loss.sum(0) / (n_valid_samples + 1e-12) # average 
                            #                                         |
                            # (eps avoids zero division error and in this case all losses will be zero anyway)
        self.state['loss'] = supervised_loss # main loss
        
        correct = ((y_hat.argmax(-1) == y.argmax(-1)) * (idx != self.padding_value)).sum(0) # correct items for classification tasks
        self.state['correct'] = correct # we can ignore this in regression cases
        
        self._fire_event('after_train_forward') # do stuff like add auxillary losses on state['loss'] 
                                                # or count sample backwards in state['per_sample_losses']
                                                
        loss = self.state['loss'] # get possibly altered loss
        
        loss.sum().backward() # grads
        
        self._fire_event('before_step') # do stuff that requires gradients but not steps (i.e. masking grads, applying LRs)
        
        self.optimizer.step()
        
        self._fire_event('after_step') # get step info (i.e. energy calculations)
        
        return loss, correct
    
    # I wasn't aware of circular dependencies and now I am
    def cleanup(self):
        print("Breaking circular references...")
        for tracker in self.trackers:
            tracker.trainer = None
        self.trackers = []
    
    # file loading and data saving is largely Gemini-Pro 2.5
    def save_checkpoint(self, path):
        """
        Saves the trainer's state to a file for later reloading.
        
        Args:
            path (str): The file path to save the checkpoint to.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tracked_data': self.state['data']  # Saves all metrics collected by observers
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """
        Loads the trainer's state from a checkpoint file. This should be
        called after initializing the Trainer but before calling train_loop.
        
        Args:
            path (str): The file path to load the checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state['data'] = checkpoint['tracked_data']
        
        # Ensure the model is on the correct device after loading
        self.model.to(self.device)
        
        print(f"Checkpoint loaded from {path}")
    
    def save_data(self, path, experimental_setup=None):
        """
        Saves just the tracked data along with experimental setup details.
    
        This is useful for logging and analysis without saving the large
        model and optimizer states.
    
        Args:
            path (str): The file path to save the data to.
            experimental_setup (dict, optional): A dictionary containing
                hyperparameters and other metadata about the run.
                Defaults to None.
        """
        if experimental_setup is None:
            experimental_setup = {}
            
        data_to_save = {
            'experimental_setup': experimental_setup,
            'tracked_data': self.state['data']
        }
        torch.save(data_to_save, path)
        print(f"Experimental data saved to {path}")
    
    def save_data_as_json(self, path, experimental_setup=None):
        """
        Saves just the tracked data along with experimental setup details
        to a human-readable JSON file.
    
        Args:
            path (str): The file path to save the JSON data to.
            experimental_setup (dict, optional): A dictionary containing
                hyperparameters and other metadata about the run.
                Defaults to None.
        """
        if experimental_setup is None:
            experimental_setup = {}
            
        data_to_save = {
            'experimental_setup': experimental_setup,
            'tracked_data': self.state['data']
        }
        
        with open(path, 'w') as f:
            json.dump(data_to_save, f, default=_json_converter, indent=4)
        print(f"Experimental data saved as JSON to {path}")
    
    @staticmethod
    def load_data(path):
        """
        Loads experimental data saved by `save_data`.
    
        This is a static method because it doesn't depend on a Trainer instance.
        You can call it directly from the class: `Trainer.load_data(path)`.
    
        Args:
            path (str): The file path to load the data from.
        
        Returns:
            dict: The loaded data, containing 'experimental_setup' and 'tracked_data'.
        """
        data = torch.load(path)
        print(f"Experimental data loaded from {path}")
        return data
    
    @staticmethod
    def load_data_from_json(path):
        """
        Loads experimental data saved by `save_data_as_json`.
    
        This is a static method that can be called directly from the class.
    
        Args:
            path (str): The file path to load the JSON data from.
        
        Returns:
            dict: The loaded data.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Experimental data loaded from JSON file: {path}")
        return data
         
    def test_loop(self):
        with torch.no_grad():
            self.model.eval()
            for (x, y, idx) in self.test_dataloader:
                x, y, idx = x.to(self.device), y.to(self.device), idx.to(self.device) 
                if len(x.shape) < 3:
                    x, y = x.unsqueeze(1), y.unsqueeze(1).repeat((1, self.n_networks, 1))
                    idx = idx.view(-1, 1)
                self._fire_event('before_test_forward')
                y_hat = self.model(x)
                self.state['test_loss'] = self.criterion(y_hat, y)
                self.state['test_accuracy'] = (y_hat.argmax(-1) == y.argmax(-1)).sum(0)
                self._fire_event('after_test_forward')
            self.model.train()
    
     
