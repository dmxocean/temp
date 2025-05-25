# scripts/sweep.py

"""
W&B Sweep runner for hyperparameter optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from typing import Dict, Any

from src.utils.manager import ConfigManager
from src.utils.logger import get_logger
from src.utils.io import load_pickle
from src.models.attention import AttentionCaptionModel
from src.models.baseline import BaselineCaptionModel
from src.preprocessing.dataset import FlickrDataset, create_data_loaders
from src.preprocessing.transforms import get_transforms
from src.preprocessing.vocabulary import Vocabulary
from src.training.trainer import Trainer

logger = get_logger(__name__)

def train_sweep(config: Dict[str, Any] = None, model_type: str = "attention"):
    """
    Train function for W&B sweep
    
    Args:
        config: Sweep configuration from W&B
        model_type: Type of model to train ("attention" or "baseline")
    """
    # Initialize wandb run within sweep
    wandb.init()
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Get configurations
    data_config = config_manager.get_data_params()
    model_config = config_manager.get_model_config(model_type)
    training_config = config_manager.get_training_params()
    
    # Override with sweep parameters
    if wandb.config:
        # Update configs with sweep parameters
        if hasattr(wandb.config, 'learning_rate'):
            training_config['learning_rate'] = wandb.config.learning_rate
        if hasattr(wandb.config, 'batch_size'):
            data_config['batch_size'] = wandb.config.batch_size
        if hasattr(wandb.config, 'hidden_size'):
            model_config['hidden_size'] = wandb.config.hidden_size
    
    # Update wandb config with all parameters
    wandb.config.update({
        "model_type": model_type,
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config
    })
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load vocabulary
        vocab_path = config_manager.paths["vocab"]
        logger.info(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
        logger.info(f"Vocabulary size: {len(vocab)}")
        
        # Get transforms
        transforms = get_transforms(data_config["image_size"])
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = FlickrDataset(
            split="train",
            vocab=vocab,
            transform=transforms["train"],
            max_seq_len=data_config["max_seq_length"],
            tokenizer_type=data_config["tokenizer"],
            captions_per_image=data_config["captions_per_image"]
        )
        
        val_dataset = FlickrDataset(
            split="val",
            vocab=vocab,
            transform=transforms["val"],
            max_seq_len=data_config["max_seq_length"],
            tokenizer_type=data_config["tokenizer"],
            captions_per_image=1  # Use single caption for validation
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, data_config
        )
        
        # Initialize model
        logger.info(f"Initializing {model_type} model...")
        if model_type == "attention":
            model = AttentionCaptionModel(
                vocab_size=len(vocab),
                **model_config
            )
        else:
            model = BaselineCaptionModel(
                vocab_size=len(vocab),
                **model_config
            )
        
        # Move to device
        model = model.to(device)
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        wandb.run.summary["total_parameters"] = total_params
        
        # Initialize optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            betas=(training_config['beta1'], training_config['beta2']),
            weight_decay=training_config['weight_decay']
        )
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config['scheduler_step_size'],
            gamma=training_config['scheduler_gamma']
        )
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
        
        # Create model paths
        os.makedirs('checkpoints', exist_ok=True)
        model_paths = {
            'checkpoint_path': f'checkpoints/{model_type}_sweep_checkpoint.pth',
            'best_model_path': f'checkpoints/{model_type}_sweep_best.pth'
        }
        
        # Initialize trainer without WandbLogger (sweep already initialized wandb)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=training_config,
            device=device,
            vocab=vocab,
            model_paths=model_paths,
            wandb_logger=None  # Will use existing wandb run
        )
        
        # Train model
        logger.info("Starting training...")
        trained_model, history = trainer.train()
        
        # Log final metrics
        if history['val_losses']:
            wandb.run.summary['best_val_loss'] = min(history['val_losses'])
        if history['bleu_scores']:
            best_bleu = max([scores.get('bleu4', 0) for scores in history['bleu_scores']])
            wandb.run.summary['best_bleu4'] = best_bleu
        
    except Exception as e:
        logger.error(f"Error during sweep training: {e}")
        raise
    finally:
        wandb.finish()

def run_sweep(model_type: str = "attention", count: int = 10):
    """
    Initialize and run W&B sweep
    
    Args:
        model_type: Type of model to train ("attention" or "baseline")
        count: Number of sweep runs to execute
    """
    # Load configurations
    config_manager = ConfigManager()
    wandb_config = config_manager.wandb_config  # Access the full wandb config including sweep
    
    # Check if sweep is enabled
    if not wandb_config.get('sweep', {}).get('enabled', False):
        logger.error("Sweep is not enabled in wandb.yaml. Set sweep.enabled to true.")
        return
    
    # Extract sweep configuration
    sweep_config = {
        "method": wandb_config['sweep']['method'],
        "metric": wandb_config['sweep']['metric'],
        "parameters": wandb_config['sweep']['parameters']
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        entity=wandb_config.get('entity'),
        project=wandb_config.get('project')
    )
    
    logger.info(f"Created sweep with ID: {sweep_id}")
    logger.info(f"Sweep URL: https://wandb.ai/{wandb_config.get('entity')}/{wandb_config.get('project')}/sweeps/{sweep_id}")
    
    # Run sweep agent
    train_fn = partial(train_sweep, model_type=model_type)
    wandb.agent(sweep_id, train_fn, count=count)
    
def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run W&B sweep for hyperparameter optimization")
    parser.add_argument(
        "--model",
        type=str,
        choices=["attention", "baseline"],
        default="attention",
        help="Model type to train"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of sweep runs to execute"
    )
    
    args = parser.parse_args()
    
    # Run sweep
    run_sweep(model_type=args.model, count=args.count)

if __name__ == "__main__":
    main()