# Libraries
from copy import deepcopy
import time
import torch
from torch_geometric.loader import DataLoader
import wandb
from tqdm.auto import tqdm
import os

from training.loss import loss_WD
from utils.dataset import use_prediction
from utils.miscellaneous import SpatialAnalysis

class Trainer(object):
    '''Training class
    -------
    optimizer: torch.optim
        model optimizer (e.g., Adam, SGD)
    loss_type: str
        options: 'RMSE', 'MAE'
    '''
    def __init__(self, optimizer, lr_scheduler=None, max_epochs=1, type_loss='RMSE',
                 report_freq: int=1, patience=3, device='cpu',
                 **training_options):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_options = training_options
        self.device = device

        self.type_loss = type_loss
        assert type_loss in ['RMSE','MAE'], "loss_type must be either 'RMSE' or 'MAE'"

        self.epoch = 0
        self.max_epochs = max_epochs
        self.report_freq = report_freq
        self.patience = patience

        #create vectors for the training and validation loss
        self.train_losses = []
        self.val_losses = []
        self.CSI_005 = []
        self.CSI_03 = []

        self.early_stop = 0
        self.best_val_loss = 1

    def _training_step(self, model, train_loader, curriculum_epoch=1, 
                      **loss_options):
        '''Function to train the model for one epoch
        ------
        model: nn.Model
            e.g., GNN model
        train_loader: DataLoader
            data loader for training dataset
        curriculum_epoch: int 
            every curriculum_epoch epochs the training window expands (default=1)
        loss_options: dict (see loss.py for more info)
            velocity_scaler: float
                weight loss for velocity terms (default=1)
        '''
        model.train()
        losses = []

        bar_freq = 0 if self.report_freq == 0 else (self.epoch) % self.report_freq == 0
        
        charge_bar = tqdm(train_loader, leave=bar_freq, disable=True)
        for batch in charge_bar:
            # reset gradients
            self.optimizer.zero_grad()

            roll_loss = []
            
            rollout_steps = self._get_rollout_steps(curriculum_epoch, batch)

            temp = batch.clone().to(self.device)

            for i in range(rollout_steps):
                # Model prediction
                preds = model(temp)
                temp.x = use_prediction(temp.x, preds, model.previous_t)

                loss = loss_WD(preds, temp.y[:,:,i], type_loss=self.type_loss, **loss_options)
                roll_loss.append(loss)

            loss = torch.stack(roll_loss).mean()
            losses.append(loss.detach())

            # Backpropagate
            loss.backward()

            # Gradient clipping (avoid exploding gradients)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

            # Update weights
            self.optimizer.step()

            charge_bar.set_description(f"Epoch {self.epoch}  Loss={loss.item():4.4f}")

        losses = torch.stack(losses).mean().cpu().item()

        return losses
        
    def fit(self, model, train_loader, val_dataset, use_progress_bar=True, **temporal_test_dataset_parameters):
        assert isinstance(train_loader, DataLoader), "Training requires train_loader to be a Dataloader object"

        has_val_dataset = val_dataset is not None and len(val_dataset) > 0

        #start measuring training time
        start_time = time.time()
        
        torch.autograd.set_detect_anomaly(True)

        if use_progress_bar:
            progress_bar = tqdm(range(self.epoch, self.max_epochs), total=self.max_epochs,
                                initial=self.epoch,leave=True, 
                                bar_format='{percentage:3.0f}%|{bar:30}| '\
                                'Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        else:
            progress_bar = range(self.epoch, self.max_epochs)

        try:
            start_epoch_time = time.time()
            for _ in progress_bar:
                self.epoch += 1

                # Model training
                train_loss = self._training_step(model, train_loader, **self.training_options)

                log_val = f"\tTrain loss = {train_loss:4.4f}"
                wandb_log_val = {"train_loss": train_loss}

                self.train_losses.append(train_loss)

                if has_val_dataset:
                    # spatial validation
                    self._get_spatial_analysis(model, val_dataset, **temporal_test_dataset_parameters)

                    # Model validation
                    val_loss = self.spatial_analyser._get_rollout_loss(type_loss=self.type_loss).mean()

                    # CSI validation
                    CSI_005 = self.spatial_analyser._get_CSI(water_threshold=0.05).mean()
                    CSI_03 = self.spatial_analyser._get_CSI(water_threshold=0.3).mean()

                    log_val += f"\tValid loss = {val_loss:1.4f}"\
                        rf"CSI_0.05 = {CSI_005:.3f}"\
                        rf"CSI_0.3 = {CSI_03:.3f}"
                    wandb_log_val.update({"valid_loss": val_loss,
                                          r"CSI_0.05": CSI_005,
                                          r"CSI_0.3": CSI_03})

                    self.val_losses.append(val_loss) 
                    self.CSI_005.append(CSI_005) 
                    self.CSI_03.append(CSI_03) 

                if use_progress_bar:
                    progress_bar.set_description(log_val)
                else:
                    print(f"Epoch: {self.epoch}\n", log_val, flush=True)
                wandb.log(wandb_log_val)

                print(f"Epoch time: {time.time()-start_epoch_time:.2f}s", flush=True)
                self._print_memory_usage(self.epoch - 1)

                self._use_learning_rate_scheduler()
                self._update_best_model(model, has_val_dataset)
                if self._early_stopping():
                    break
        except KeyboardInterrupt:
            self.epoch -= 1

        self.training_time = time.time() - start_time

        print(f"Training time: {self.training_time:.4f}s", flush=True)
        final_wandb_log = {"training_time": self.training_time}
        if has_val_dataset:
            min_val_loss = torch.tensor(self.val_losses).min()
            argmin_val_loss = torch.tensor(self.val_losses).argmin()
            final_wandb_log.update({"valid_loss": min_val_loss,
                   r"CSI_0.05": self.CSI_005[argmin_val_loss],
                   r"CSI_0.3": self.CSI_03[argmin_val_loss]})
        wandb.log(final_wandb_log)

        try:
            print("Loading best model...")
            model.load_state_dict(self.best_model)
        except:
            pass #avoid blocking processes even if loss is very bad

    def _save_model(self, model, model_name="best_model.h5", save_dir=None):
        '''Save model in directory'''
        if save_dir is None:
            save_dir = wandb.run.dir
        save_dir = os.path.join(save_dir, model_name)
        torch.save(model.state_dict(), save_dir)

    def _load_model(self, model, model_name="best_model.h5", save_dir=None):
        '''Load model from directory'''
        if save_dir is None:
            save_dir = wandb.run.dir
        save_dir = os.path.join(save_dir, model_name)
        model.load_state_dict(torch.load(save_dir, map_location=self.device))

    def _get_rollout_steps(self, curriculum_epoch, batch):
        '''Returns number of rollout steps/size of the training window'''
        if curriculum_epoch == 0:
            rollout_steps = batch.y.shape[-1]
        else:
            rollout_steps = self.epoch//curriculum_epoch+1
            if rollout_steps > batch.y.shape[-1]:
                rollout_steps = batch.y.shape[-1]
        self.current_rollout_steps = rollout_steps
        return rollout_steps

    def _early_stopping(self):
        '''Stop training if validation keeps increasing'''
        should_stop = False

        if len(self.val_losses) < 3:
            pass
        elif self.val_losses[-1]>=self.val_losses[-2]:
            self.early_stop += 1
        else:
            self.early_stop = 0
        
        if self.early_stop == self.patience:
            print("Early stopping! Epoch:", self.epoch)
            should_stop = True

        return should_stop

    def _use_learning_rate_scheduler(self):
        '''If present, use a learning rate scheduler step'''
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _update_best_model(self, model, has_val_dataset):
        '''Saves the model with best validation loss'''
        if has_val_dataset:
            if self.val_losses[-1] < self.best_val_loss:
                self.best_val_loss = self.val_losses[-1]
                self.best_model = deepcopy(model.state_dict())
        else:
            self.best_model = deepcopy(model.state_dict())

    def _get_spatial_analysis(self, model, val_dataset, **temporal_test_dataset_parameters):
        self.spatial_analyser = SpatialAnalysis(model, val_dataset, self.device, **temporal_test_dataset_parameters)

    def _print_memory_usage(self, epoch: int):
        def bytes_to_gb(b: float) -> float:
            return round(b / (1024 ** 3), 2)

        print(f'Usage Statistics (Epoch {epoch+1}): ', flush=True)
        gpu_usage = torch.cuda.mem_get_info()
        free = bytes_to_gb(gpu_usage[0])
        total = bytes_to_gb(gpu_usage[1])
        print(f"\tGPU Usage: {free}GB / {total}GB", flush=True)

        gpu_allocated = torch.cuda.memory_allocated()
        gpu_cached = torch.cuda.memory_reserved()
        print(f"\tCUDA GPU Allocated: {bytes_to_gb(gpu_allocated)}GB", flush=True)
        print(f"\tCUDA GPU Cached: {bytes_to_gb(gpu_cached)}GB", flush=True)
