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
                    # Learning rate scheduling
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.train_metrics['learning_rates'].append(current_lr)
                    self.scheduler.step()
                
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
    Pretraining trainer for modality alignment tasks.
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
    Classification trainer for supervised learning tasks.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        adv_loss_fn: Optional[nn.Module] = None,
        lambda_adv: float = 0.1,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None,
        use_adversarial: bool = False
    ):
        """
        Initialize classification trainer with optional adversarial training.
        
        Args:
            model (nn.Module): The neural network model to train.
            optimizer (Optimizer): Optimizer for model parameters.
            loss_fn (nn.Module): Loss function for the main task.
            adv_loss_fn (Optional[nn.Module]): Loss function for the adversarial task.
            lambda_adv (float): Weight for the adversarial loss component.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
            device (Optional[str]): Compute device (cuda/mps/cpu).
            use_adversarial (bool): Whether to use adversarial training.
        """
        super().__init__(model, optimizer, loss_fn, scheduler, device)
        
        # Adversarial training components
        self.adv_loss_fn = adv_loss_fn
        self.lambda_adv = lambda_adv
        self.use_adversarial = use_adversarial
        
        # Additional metrics for adversarial training
        if self.use_adversarial:
            self.train_metrics.update({
                'main_losses': [],
                'adv_losses': []
            })
    
    
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
        sex_data = batch["sex"]
        input_ = (cell_type_list, gene_data, sex_data)
        
        # Forward pass
        if self.use_adversarial:
            logits, adv_logits = self.model(input_)
            
            # Compute main task loss
            labels = batch["labels"]
            main_loss = self.loss_fn(logits, labels)
            
            # Compute adversarial loss
            adv_loss = self.adv_loss_fn(adv_logits.squeeze(), sex_data.float())
            
            # Total loss (note the negative sign for adversarial loss)
            # We want to maximize adversarial loss (minimize its negative)
            loss = main_loss - self.lambda_adv * adv_loss
            
            result = {
                'loss': loss,
                'main_loss': main_loss,
                'adv_loss': adv_loss,
                'logits': logits,
                'adv_logits': adv_logits
            }
        else:
            # Standard forward pass without adversarial component
            output = self.model(input_)
            
            # Compute loss
            labels = batch["labels"]
            loss = self.loss_fn(output, labels)
            
            result = {
                'loss': loss,
                'output': output
            }
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Track additional metrics for adversarial training
        if self.use_adversarial:
            self.train_metrics['main_losses'].append(main_loss.item())
            self.train_metrics['adv_losses'].append(adv_loss.item())
        
        return result


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
            sex_data = batch["sex"]
            input_ = (cell_type_list, gene_data, sex_data)
            labels = batch["labels"]

            if self.use_adversarial:
                logits, adv_logits = self.model(input_)
                main_loss = self.loss_fn(logits, labels)
                adv_loss = self.adv_loss_fn(adv_logits.squeeze(), sex_data.float())
                loss = main_loss - self.lambda_adv * adv_loss
                
                result = {
                    'loss': loss,
                    'main_loss': main_loss,
                    'adv_loss': adv_loss,
                    'logits': logits,
                    'adv_logits': adv_logits
                }
            else:
                output = self.model(input_)
                loss = self.loss_fn(output, labels)
                
                result = {
                    'loss': loss,
                    'output': output
                }
        
        return result


    def predict(self, test_loader: DataLoader) -> tuple:
        """
        Run prediction on a data loader and return predictions with labels.
        
        Args:
            test_loader (DataLoader): Data loader for prediction.
        
        Returns:
            tuple: (predictions, true_labels, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Prepare input
                cell_type_list = batch["cell_type"]
                gene_data = batch["geneformer_embeddings"]
                sex_data = batch["sex"]
                input_ = (cell_type_list, gene_data, sex_data)
                
                # Forward pass
                if self.use_adversarial:
                    logits, _ = self.model(input_)  # Ignore adversarial output
                else:
                    logits = self.model(input_)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Store results
                all_preds.append(preds.cpu())
                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu())
        
        # Concatenate results
        predictions = torch.cat(all_preds).numpy()
        
        if all_labels:
            labels = torch.cat(all_labels).numpy()
            return predictions, labels
        else:
            return predictions
    

    def plot_training_metrics(self, save_path: Optional[str] = None):
        """
        Plot training metrics with adversarial components if enabled.
        
        Args:
            save_path (Optional[str]): Path to save the plot.
        """
        if self.use_adversarial:
            plt.figure(figsize=(15, 8))
            
            # Main loss plot
            plt.subplot(231)
            plt.plot(self.train_metrics['step_losses'])
            plt.title('Combined Step Losses')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            
            # Main task loss
            plt.subplot(232)
            plt.plot(self.train_metrics['main_losses'])
            plt.title('Main Task Losses')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            
            # Adversarial loss
            plt.subplot(233)
            plt.plot(self.train_metrics['adv_losses'])
            plt.title('Adversarial Losses')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            
            # Epoch losses
            plt.subplot(234)
            plt.plot(self.train_metrics['epoch_losses'])
            plt.title('Epoch Losses')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            
            # Learning rates
            if self.train_metrics['learning_rates']:
                plt.subplot(235)
                plt.plot(self.train_metrics['learning_rates'])
                plt.title('Learning Rates')
                plt.xlabel('Steps')
                plt.ylabel('LR')
        else:
            # Use the parent class implementation for standard training
            super().plot_training_metrics(save_path)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
        
        plt.show()
    

class VanillaTrainer(TrainerBase):
    """
    Standard trainer for supervised learning tasks.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize a vanilla classification trainer to test the model without modality alignment.
        
        Args:
            model (nn.Module): The neural network model to train.
            optimizer (Optimizer): Optimizer for model parameters.
            loss_fn (nn.Module): Loss function for the main task.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
            device (Optional[str]): Compute device (cuda/mps/cpu).
        """
        super().__init__(model, optimizer, loss_fn, scheduler, device)
        
    
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
        sex_data = batch["sex"]
        input_ = (cell_type_list, gene_data, sex_data)
        
        # Standard forward pass without adversarial component
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
            sex_data = batch["sex"]
            input_ = (cell_type_list, gene_data, sex_data)
            labels = batch["labels"]
            output = self.model(input_)
            loss = self.loss_fn(output, labels)                
        
        return {
                'loss': loss,
                'output': output
            }


    def predict(self, test_loader: DataLoader) -> tuple:
        """
        Run prediction on a data loader and return predictions with labels.
        
        Args:
            test_loader (DataLoader): Data loader for prediction.
        
        Returns:
            tuple: (predictions, true_labels, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Prepare input
                cell_type_list = batch["cell_type"]
                gene_data = batch["geneformer_embeddings"]
                sex_data = batch["sex"]
                input_ = (cell_type_list, gene_data, sex_data)
                
                # Forward pass
                if self.use_adversarial:
                    logits, _ = self.model(input_)  # Ignore adversarial output
                else:
                    logits = self.model(input_)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Store results
                all_preds.append(preds.cpu())
                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu())
        
        # Concatenate results
        predictions = torch.cat(all_preds).numpy()
        
        if all_labels:
            labels = torch.cat(all_labels).numpy()
            return predictions, labels
        else:
            return predictions
    

