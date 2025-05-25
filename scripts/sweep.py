# scripts/sweep.py

"""
W&B Sweep runner for hyperparameter optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
from functools import partial
from typing import Dict, Any

from src.utils.manager import ConfigManager
from src.utils.logger import get_logger
from src.models.attention import AttentionModel
from src.models.baseline import BaselineModel
from src.preprocessing.dataset import get_data_loaders
from src.training.trainer import Trainer
from src.utils.wanlog import WandbLogger

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
    
    # Load base configurations
    config_manager = ConfigManager()
    data_config = config_manager.load_config("data")
    model_config = config_manager.load_config("model")
    training_config = config_manager.load_config("training")
    
    # Override with sweep parameters
    if wandb.config:
        # Update configs with sweep parameters
        if hasattr(wandb.config, 'learning_rate'):
            training_config['learning_rate'] = wandb.config.learning_rate
        if hasattr(wandb.config, 'batch_size'):
            data_config['batch_size'] = wandb.config.batch_size
        if hasattr(wandb.config, 'hidden_size'):
            if model_type == "attention":
                model_config['attention']['decoder_hidden_size'] = wandb.config.hidden_size
            else:
                model_config['baseline']['hidden_size'] = wandb.config.hidden_size
    
    # Update wandb config with all parameters
    wandb.config.update({
        "model_type": model_type,
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config
    })
    
    try:
        # Get data loaders
        logger.info("Loading data...")
        train_loader, val_loader, vocab = get_data_loaders(data_config)
        
        # Initialize model
        logger.info(f"Initializing {model_type} model...")
        if model_type == "attention":
            model = AttentionModel(
                vocab_size=len(vocab),
                **model_config['attention']
            )
        else:
            model = BaselineModel(
                vocab_size=len(vocab),
                **model_config['baseline']
            )
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        wandb.run.summary["total_parameters"] = total_params
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            betas=(training_config['beta1'], training_config['beta2']),
            weight_decay=training_config['weight_decay']
        )
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config['scheduler_step_size'],
            gamma=training_config['scheduler_gamma']
        )
        
        # Initialize criterion
        criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
        
        # Create model paths
        model_paths = {
            'checkpoint': f'checkpoints/{model_type}_sweep_checkpoint.pth',
            'best': f'checkpoints/{model_type}_sweep_best.pth'
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
            best_bleu = max([scores.get('bleu-4', 0) for scores in history['bleu_scores']])
            wandb.run.summary['best_bleu4'] = best_bleu
        
    except Exception as e:
        logger.error(f"Error during sweep training: {e}")
        raise
    finally:
        wandb.finish()

def run_sweep(model_type: str = "attention"):
    """
    Initialize and run W&B sweep
    
    Args:
        model_type: Type of model to train ("attention" or "baseline")
    """
    # Load configurations
    config_manager = ConfigManager()
    wandb_config = config_manager.load_config("wandb")
    
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
        entity=wandb_config['entity'],
        project=wandb_config['project']
    )
    
    logger.info(f"Created sweep with ID: {sweep_id}")
    logger.info(f"Sweep URL: https://wandb.ai/{wandb_config['entity']}/{wandb_config['project']}/sweeps/{sweep_id}")
    
    # Run sweep agent
    train_fn = partial(train_sweep, model_type=model_type)
    wandb.agent(sweep_id, train_fn, count=10)  # Run 10 sweep runs
    
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
    run_sweep(model_type=args.model)

if __name__ == "__main__":
    main()