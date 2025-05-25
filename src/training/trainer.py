# src/training/trainer.py

"""
Training loop implementation for image captioning models
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .metrics import calculate_loss, calculate_bleu
from ..utils.logger import VerboseLogger, AverageMeter
from ..utils.io import save_checkpoint, check_model_availability
from ..utils.wanlog import WandbLogger
from ..preprocessing.vocabulary import Vocabulary

class Trainer:
    """Unified trainer for both baseline and attention models"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: nn.Module, config: Dict, device: torch.device, vocab: Vocabulary,
                 model_paths: Dict, wandb_logger: Optional[WandbLogger] = None):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss criterion
            config: Training configuration
            device: Device to train on
            vocab: Vocabulary object
            model_paths: Dictionary with checkpoint and best model paths
            wandb_logger: Optional wandb logger
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        self.vocab = vocab
        self.model_paths = model_paths
        self.wandb_logger = wandb_logger
        
        # Extract configuration parameters
        self.num_epochs = config["num_epochs"]
        self.pad_idx = vocab.stoi["<PAD>"]
        self.print_frequency = config["print_frequency"]
        self.eval_every = config["eval_every"]
        self.bleu_every = config["bleu_every"]
        self.max_bleu_samples = config["max_bleu_samples"]
        self.early_stopping_patience = config["early_stopping_patience"]
        self.clip_grad_norm = config["clip_grad_norm"]
        
        # Determine model type
        self.is_attention_model = hasattr(model, 'caption_image_with_attention')
        self.model_name = "Attention" if self.is_attention_model else "Baseline"
        
        # Initialize logger
        self.logger = VerboseLogger(self.print_frequency)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_bleu = 0
        self.patience_counter = 0
        self.history = {
            'epochs': [],
            'train_losses': [],
            'val_epochs': [],
            'val_losses': [],
            'bleu_epochs': [],
            'bleu_scores': []
        }
        
        # Global step counter for wandb
        self.global_step = 0
    
    def train_epoch(self) -> float:
        """
        Train model for one epoch
        
        Returns:
            avg_loss: Average training loss
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        epoch_loss = 0
        total_tokens = 0
        batch_time = AverageMeter()
        
        # Get start time
        start_batch_time = time.time()
        
        # Iterate over batches with progress bar
        with tqdm(total=len(self.train_loader), desc="Training") as pbar:
            for i, (images, captions, lengths) in enumerate(self.train_loader):
                # Move to device
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # Forward pass - different for baseline vs attention
                if self.is_attention_model:
                    outputs, alphas = self.model(images, captions, lengths)
                    
                    # Prepare targets (shift by one)
                    targets = captions[:, 1:]  # Remove < SOS >
                    outputs = outputs[:, :-1, :]  # Remove last prediction
                    
                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, outputs.size(2))
                    targets = targets.reshape(-1)
                    
                    # Calculate loss
                    loss, n_tokens = calculate_loss(outputs, targets, self.criterion, self.pad_idx)
                    
                    # Add attention regularization (encourage diversity in attention)
                    alpha_c = 1.0  # Attention regularization factor
                    att_reg = alpha_c * ((1 - alphas.sum(dim=1)) ** 2).mean()
                    total_loss = loss + att_reg
                else:
                    outputs = self.model(images, captions, lengths)
                    
                    # Prepare targets (shift by one)
                    targets = captions[:, 1:]  # Remove < SOS >
                    outputs = outputs[:, :-1, :]  # Remove last prediction
                    
                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, outputs.size(2))
                    targets = targets.reshape(-1)
                    
                    # Calculate loss
                    loss, n_tokens = calculate_loss(outputs, targets, self.criterion, self.pad_idx)
                    total_loss = loss
                
                # Update metrics
                epoch_loss += total_loss.item() * n_tokens
                total_tokens += n_tokens
                
                # Update batch time
                batch_time.update(time.time() - start_batch_time)
                start_batch_time = time.time()
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                
                # Update weights
                self.optimizer.step()
                
                # Log to wandb
                if self.wandb_logger:
                    self.wandb_logger.log_batch(i, total_loss.item(), self.global_step)
                
                # Update global step
                self.global_step += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "time/batch": f"{batch_time.avg:.3f}s"})
                
                # Print progress
                self.logger.log_batch(i, len(self.train_loader), total_loss.item(), batch_time.avg)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / total_tokens if total_tokens > 0 else float('inf')
        
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate model
        
        Returns:
            avg_loss: Average validation loss
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        epoch_loss = 0
        total_tokens = 0
        
        # No gradient calculation needed
        with torch.no_grad():
            # Iterate over batches with progress bar
            with tqdm(total=len(self.val_loader), desc="Validation") as pbar:
                for images, captions, lengths in self.val_loader:
                    # Move to device
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    
                    # Forward pass - different for baseline vs attention
                    if self.is_attention_model:
                        outputs, alphas = self.model(images, captions, lengths)
                        
                        # Prepare targets (shift by one)
                        targets = captions[:, 1:]  # Remove < SOS >
                        outputs = outputs[:, :-1, :]  # Remove last prediction
                        
                        # Reshape for loss calculation
                        outputs = outputs.reshape(-1, outputs.size(2))
                        targets = targets.reshape(-1)
                        
                        # Calculate loss
                        loss, n_tokens = calculate_loss(outputs, targets, self.criterion, self.pad_idx)
                        
                        # Add attention regularization
                        alpha_c = 1.0  # Attention regularization factor
                        att_reg = alpha_c * ((1 - alphas.sum(dim=1)) ** 2).mean()
                        total_loss = loss + att_reg
                    else:
                        outputs = self.model(images, captions, lengths)
                        
                        # Prepare targets (shift by one)
                        targets = captions[:, 1:]  # Remove < SOS >
                        outputs = outputs[:, :-1, :]  # Remove last prediction
                        
                        # Reshape for loss calculation
                        outputs = outputs.reshape(-1, outputs.size(2))
                        targets = targets.reshape(-1)
                        
                        # Calculate loss
                        loss, n_tokens = calculate_loss(outputs, targets, self.criterion, self.pad_idx)
                        total_loss = loss
                    
                    # Update metrics
                    epoch_loss += total_loss.item() * n_tokens
                    total_tokens += n_tokens
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / total_tokens if total_tokens > 0 else float('inf')
        
        return avg_loss
    
    def train(self) -> Tuple[nn.Module, Dict]:
        """
        Full training loop
        
        Returns:
            model: Trained model
            history: Training history
        """
        # Check if a trained model exists
        model_status, model_checkpoint = check_model_availability(
            self.model_paths["checkpoint_path"], 
            self.model_paths["best_model_path"]
        )
        
        # If a trained model exists, load it
        if model_status == "trained":
            print(f"Found trained {self.model_name} model, loading weights...")
            self.model.load_state_dict(model_checkpoint['state_dict'])
            
            # Load training history if available
            if 'training_history' in model_checkpoint:
                self.history = model_checkpoint['training_history']
                print("Loaded training history")
            
            print(f"Skipping training for {self.model_name} model")
            return self.model, self.history
        
        # If a checkpoint exists, resume training
        elif model_status == "checkpoint":
            print(f"Found checkpoint for {self.model_name} model, resuming training...")
            
            # Load model state
            self.model.load_state_dict(model_checkpoint['state_dict'])
            
            # Load optimizer and scheduler states if available
            if 'optimizer' in model_checkpoint:
                self.optimizer.load_state_dict(model_checkpoint['optimizer'])
                print("Loaded optimizer state")
            
            if 'scheduler' in model_checkpoint and self.scheduler:
                self.scheduler.load_state_dict(model_checkpoint['scheduler'])
                print("Loaded scheduler state")
            
            # Load training history if available
            if 'training_history' in model_checkpoint:
                self.history = model_checkpoint['training_history']
                print("Loaded training history")
            
            # Get starting epoch
            start_epoch = model_checkpoint.get('epoch', 0)
            self.best_val_loss = model_checkpoint.get('val_loss', float('inf'))
            print(f"Resuming training from epoch {start_epoch + 1}...")
            
        else:
            # Start fresh training
            print(f"Starting fresh training for {self.model_name} model...")
            start_epoch = 0
        
        # Log model summary to wandb
        if self.wandb_logger:
            self.wandb_logger.log_model_summary(self.model)
            self.wandb_logger.update_config({
                "model_name": self.model_name,
                **self.config
            })
        
        # Start training
        self.logger.start_training(self.num_epochs - start_epoch, self.model_name)
        
        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            epoch_num = epoch + 1  # 1-based epoch numbering
            
            # Start epoch
            self.logger.start_epoch(epoch_num, self.num_epochs)
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Store train loss
            self.history['epochs'].append(epoch_num)
            self.history['train_losses'].append(train_loss)
            
            # Validation (based on eval_every)
            should_validate = epoch_num % self.eval_every == 0 or epoch_num == self.num_epochs
            if should_validate:
                # Validate
                val_loss = self.validate()
                
                # Store validation results
                self.history['val_epochs'].append(epoch_num)
                self.history['val_losses'].append(val_loss)
                
                # Check for new best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    print(f"New best model with validation loss: {val_loss:.4f}")
                    save_checkpoint({
                        'epoch': epoch_num,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                        'val_loss': val_loss,
                        'config': self.config,
                        'training_history': self.history
                    }, filepath=self.model_paths["best_model_path"], is_best=True)
                else:
                    self.patience_counter += 1
                
                # Print validation results
                self.logger.log_epoch_end(epoch_num, train_loss, val_loss)
                
                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step(val_loss)
            else:
                # Print training results only
                self.logger.log_epoch_end(epoch_num, train_loss)
            
            # Calculate BLEU scores (based on bleu_every)
            should_calculate_bleu = epoch_num % self.bleu_every == 0 or epoch_num == self.num_epochs
            if should_calculate_bleu:
                # Calculate BLEU scores
                print("Calculating BLEU scores...")
                bleu = calculate_bleu(self.model, self.val_loader, self.vocab, 
                                    self.device, max_samples=self.max_bleu_samples)
                
                # Store BLEU scores
                self.history['bleu_epochs'].append(epoch_num)
                self.history['bleu_scores'].append(bleu)
                
                # Check for new best model (based on BLEU-4)
                if bleu['bleu4'] > self.best_bleu:
                    self.best_bleu = bleu['bleu4']
                
                # Print BLEU scores
                self.logger.log_bleu_scores(bleu)
            
            # Log to wandb
            if self.wandb_logger:
                log_dict = {"epoch": epoch_num, "train_loss": train_loss}
                if should_validate:
                    log_dict["val_loss"] = val_loss
                if should_calculate_bleu:
                    log_dict.update(bleu)
                self.wandb_logger.log_epoch(epoch_num, train_loss, 
                                          val_loss if should_validate else None,
                                          bleu if should_calculate_bleu else None)
            
            # Save regular checkpoint
            save_checkpoint({
                'epoch': epoch_num,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss if should_validate else None,
                'config': self.config,
                'training_history': self.history
            }, filepath=self.model_paths["checkpoint_path"])
            
            # Check for early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch_num} epochs")
                break
        
        # Training completed
        self.logger.log_training_end()
        
        # Load the best model
        print(f"Loading best {self.model_name} model...")
        best_checkpoint = torch.load(self.model_paths["best_model_path"])
        self.model.load_state_dict(best_checkpoint['state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation loss {best_checkpoint['val_loss']:.4f}")
        
        return self.model, self.history