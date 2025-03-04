from utils.loss import ContrastiveLoss

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Optional, Union, Callable, Dict, Any
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler



class TrainerBase:
    """
    Base trainer class providing a flexible and extensible training framework.
    
    Supports:
    - Standard and adversarial training
    - Device management
    - Learning rate scheduling 
    - Logging and metrics tracking
    - Customizable training loops
    """
    def __init__(
        self, 
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer with core training components.

        Args:
            model (nn.Module): The neural network model to train.
            optimizer (Optimizer): Optimizer for model parameters.
            loss_fn (Optional[nn.Module]): Loss function for training.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
            device (Optional[str]): Compute device (cuda/mps/cpu).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        # Device management
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Training metrics tracking
        self.train_metrics = {
            'epoch_losses': [],
            'step_losses': [],
            'learning_rates': []
        }
        self.validation_metrics = []


    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Default training step to be overridden by subclasses.
        
        Args:
            batch (Dict[str, Any]): A batch of training data.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss and other metrics.
        """
        raise NotImplementedError("Subclasses must implement train_step method")


    def train(
        self, 
        train_loader: DataLoader, 
        num_epochs: int = 10, 
        val_loader: Optional[DataLoader] = None,
        logging_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Main training loop with advanced features.
        
        Args:
            train_loader (DataLoader): Training data loader.
            num_epochs (int): Number of training epochs.
            val_loader (Optional[DataLoader]): Validation data loader.
            logging_steps (int): Frequency of logging during training.
        
        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        # Prepare model for training
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Epoch progress bar
            progress_bar = tqdm(
                enumerate(train_loader), 
                total=len(train_loader), 
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for step, batch in progress_bar:
                # Prepare batch
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Training step
                step_result = self.train_step(batch)
                loss = step_result['loss']
                
                # Update metrics
                epoch_loss += loss.item()
                self.train_metrics['step_losses'].append(loss.item())
                
                # Optional learning rate tracking
                if self.scheduler:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.train_metrics['learning_rates'].append(current_lr)
                
                # Logging and progress bar
                if step % logging_steps == 0:
                    progress_bar.set_postfix({'loss': loss.item()})


            # Epoch summary
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_metrics['epoch_losses'].append(avg_epoch_loss)
            
            # Optional validation based on validation loader
            if val_loader is not None:
                val_metrics = self.validate(val_loader) # return a dict containing val metrics (loss, ..)
                self.validation_metrics.append(val_metrics)


            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
        
        metrics = {
            'train': self.train_metrics,
        }
        if self.validation_metrics:
            metrics['validation'] = self.validation_metrics  # list of dicts
        
        return metrics


    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Optional validation method.
        
        Args:
            val_loader (DataLoader): Validation data loader.
        
        Returns:
            Dict[str, float]: Validation metrics.
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                val_step_result = self.validate_step(batch)
                val_loss += val_step_result['loss'].item()
        
        return {
            'val_loss': val_loss / len(val_loader)
        }


    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Default validation step to be overridden by subclasses.
        
        Args:
            batch (Dict[str, Any]): A batch of validation data.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing validation loss and metrics.
        """
        raise NotImplementedError("Subclasses must implement validate_step method")


    def plot_training_metrics(self, save_path: Optional[str] = None):
        """
        Plot training metrics.
        
        Args:
            save_path (Optional[str]): Path to save the plot.
        """
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(131)
        plt.plot(self.train_metrics['step_losses'])
        plt.title('Step Losses')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        
        # Epoch losses
        plt.subplot(132)
        plt.plot(self.train_metrics['epoch_losses'])
        plt.title('Epoch Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        # Learning rates
        if self.train_metrics['learning_rates']:
            plt.subplot(133)
            plt.plot(self.train_metrics['learning_rates'])
            plt.title('Learning Rates')
            plt.xlabel('Steps')
            plt.ylabel('LR')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


    def save_model(self, path: str):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))


class PretrainTrainer(TrainerBase):
    """
    Standard trainer for supervised learning tasks.
    """
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Training step for standard supervised learning.
        
        Args:
            batch (Dict[str, Any]): Batch of training data.
        
        Returns:
            Dict[str, torch.Tensor]: Training step results.
        """
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Prepare input
        cell_type_list = batch["cell_type"]
        gene_data = batch["geneformer_embeddings"]
        input_ = (cell_type_list, gene_data)
        
        # Forward pass
        output = self.model(input_)
        
        # Compute loss
        loss = self.loss_fn(*output)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss,
            'output': output
        }


    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Validation step for standard supervised learning.
        
        Args:
            batch (Dict[str, Any]): Batch of validation data.
        
        Returns:
            Dict[str, torch.Tensor]: Validation step results.
        """
        with torch.no_grad():
            cell_type_list = batch["cell_type"]
            gene_data = batch["geneformer_embeddings"]
            input_ = (cell_type_list, gene_data)
            output = self.model(input_)
            loss = self.loss_fn(*output)
        
        return {
            'loss': loss,
            'output': output
        }


class ClassificationTrainer(TrainerBase):
    """
    Standard trainer for supervised learning tasks.
    """
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Training step for standard supervised learning.
        
        Args:
            batch (Dict[str, Any]): Batch of training data.
        
        Returns:
            Dict[str, torch.Tensor]: Training step results.
        """
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Prepare input
        cell_type_list = batch["cell_type"]
        gene_data = batch["geneformer_embeddings"]
        input_ = (cell_type_list, gene_data, batch["sex"])
        
        # Forward pass
        output = self.model(input_)
        
        # Compute loss
        labels = batch["labels"]
        loss = self.loss_fn(output, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss,
            'output': output
        }

    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Validation step for standard supervised learning.
        
        Args:
            batch (Dict[str, Any]): Batch of validation data.
        
        Returns:
            Dict[str, torch.Tensor]: Validation step results.
        """
        with torch.no_grad():
            cell_type_list = batch["cell_type"]
            gene_data = batch["geneformer_embeddings"]
            input_ = (cell_type_list, gene_data, batch["sex"])

            output = self.model(input_)
            labels = batch["labels"]
            loss = self.loss_fn(output, labels)
        
        return {
            'loss': loss,
            'output': output
        }
  
    
class AdversarialTrainer(TrainerBase):
    """
    Adversarial trainer for domain adaptation and fair representation learning.
    """
    def __init__(
        self, 
        model: nn.Module,
        optimizer_main: Optimizer,
        optimizer_adv: Optimizer,
        criterion_main: nn.Module,
        criterion_adv: nn.Module,
        lambda_adv: float = 1.0,
        scheduler_main: Optional[_LRScheduler] = None,
        scheduler_adv: Optional[_LRScheduler] = None,
        device: Optional[str] = None
    ):
        """
        Initialize adversarial trainer with main and adversarial components.
        
        Args:
            model (nn.Module): Model with main and adversarial components.
            optimizer_main (Optimizer): Optimizer for main task.
            optimizer_adv (Optimizer): Optimizer for adversarial task.
            criterion_main (nn.Module): Loss for main task.
            criterion_adv (nn.Module): Loss for adversarial task.
            lambda_adv (float): Adversarial loss weight.
            scheduler_main (Optional[_LRScheduler]): Scheduler for main optimizer.
            scheduler_adv (Optional[_LRScheduler]): Scheduler for adversary optimizer.
            device (Optional[str]): Compute device.
        """
        super().__init__(model, optimizer_main, criterion_main, scheduler_main, device)
        self.optimizer_adv = optimizer_adv
        self.criterion_adv = criterion_adv
        self.lambda_adv = lambda_adv
        self.scheduler_adv = scheduler_adv

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Adversarial training step.
        
        Args:
            batch (Dict[str, Any]): Batch of training data.
        
        Returns:
            Dict[str, torch.Tensor]: Training step results.
        """
        # Prepare inputs
        x_batch = batch["geneformer_embeddings"]
        y_batch = batch["donor_id"]
        sex_batch = batch["sex"]
        cell_type_list = batch["cell_type"]
        input_ = (cell_type_list, x_batch, sex_batch)

        # Step 1: Update adversary
        self.optimizer_adv.zero_grad()
        logits, adversary_logits = self.model(input_, reverse_lambda=1.0)
        loss_adv = self.criterion_adv(adversary_logits.squeeze(), sex_batch.float())
        loss_adv.backward()
        self.optimizer_adv.step()
        
        # Step 2: Update main model
        self.optimizer.zero_grad()
        logits, adversary_logits = self.model(input_, reverse_lambda=1.0)
        loss_main = self.criterion_main(logits, y_batch)
        loss_adv = self.criterion_adv(adversary_logits.squeeze(), sex_batch.float())
        loss = loss_main - self.lambda_adv * loss_adv
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss,
            'main_loss': loss_main,
            'adv_loss': loss_adv,
            'logits': logits,
            'adversary_logits': adversary_logits
        }

    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Validation step for adversarial training.
        
        Args:
            batch (Dict[str, Any]): Batch of validation data.
        
        Returns:
            Dict[str, torch.Tensor]: Validation step results.
        """
        with torch.no_grad():
            x_batch = batch["geneformer_embeddings"]
            y_batch = batch["donor_id"]
            sex_batch = batch["sex"]
            cell_type_list = batch["cell_type"]
            input_ = (cell_type_list, x_batch, sex_batch)

            logits, adversary_logits = self.model(input_, reverse_lambda=1.0)
            loss_main = self.criterion_main(logits, y_batch)
            loss_adv = self.criterion_adv(adversary_logits.squeeze(), sex_batch.float())
            loss = loss_main - self.lambda_adv * loss_adv

        return {
            'loss': loss,
            'main_loss': loss_main,
            'adv_loss': loss_adv,
            'logits': logits,
            'adversary_logits': adversary_logits
        }
