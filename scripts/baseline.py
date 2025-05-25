# scripts/baseline.py

"""
Training script for baseline CNN-LSTM model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.manager import ConfigManager
from src.utils.io import load_pickle, save_json
from src.utils.wanlog import WandbLogger
from src.preprocessing.vocabulary import Vocabulary
from src.preprocessing.dataset import FlickrDataset, create_data_loaders, create_debug_loader
from src.preprocessing.transforms import get_transforms
from src.models.baseline import BaselineCaptionModel
from src.training.trainer import Trainer
from src.training.metrics import calculate_bleu
from src.visualization.captioning import plot_training_history, visualize_sample_captions

def main():
    """Main training function for baseline model"""
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Get configurations
    data_config = config_manager.get_data_params()
    model_config = config_manager.get_model_config("baseline")
    training_config = config_manager.get_training_params()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab_path = config_manager.paths["vocab"]
    print(f"\nLoading vocabulary from {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load data splits
    splits_path = config_manager.paths["splits"]
    print(f"\nLoading data splits from {splits_path}")
    splits = load_pickle(splits_path)
    train_df = splits['train']
    val_df = splits['val']
    test_df = splits['test']
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Get transforms
    transform_train, transform_val = get_transforms(
        resize=data_config["image"]["resize_size"],
        crop=data_config["image"]["crop_size"]
    )
    
    # Create datasets
    images_dir = data_config["dataset"]["images_dir"]
    
    train_dataset = FlickrDataset(
        data_df=train_df,
        root_dir=images_dir,
        vocab=vocab,
        transform=transform_train
    )
    
    val_dataset = FlickrDataset(
        data_df=val_df,
        root_dir=images_dir,
        vocab=vocab,
        transform=transform_val
    )
    
    test_dataset = FlickrDataset(
        data_df=test_df,
        root_dir=images_dir,
        vocab=vocab,
        transform=transform_val
    )
    
    # Create data loaders
    batch_size = training_config["batch_size"]
    
    if config_manager.debug:
        print("\nDEBUG MODE: Creating debug loaders")
        train_loader = create_debug_loader(train_dataset, batch_size, 100, vocab)
        val_loader = create_debug_loader(val_dataset, batch_size, 50, vocab)
        test_loader = create_debug_loader(test_dataset, batch_size, 50, vocab)
    else:
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size, vocab
        )
    
    print(f"\nDataLoader batches:")
    print(f"Train: {len(train_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    
    # Initialize model
    print("\nInitializing baseline model...")
    model = BaselineCaptionModel(
        embed_size=model_config["embed_size"],
        hidden_size=model_config["hidden_size"],
        vocab_size=len(vocab),
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Create loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=training_config["lr_scheduler_factor"],
        patience=training_config["lr_scheduler_patience"],
        verbose=True
    )
    
    # Get model paths
    model_paths = {
        "checkpoint_path": os.path.join(config_manager.get_model_dir("baseline"), "checkpoint.pth"),
        "best_model_path": os.path.join(config_manager.get_model_dir("baseline"), "best_model.pth")
    }
    
    # Initialize wandb logger
    wandb_config = config_manager.get_wandb_params()
    if wandb_config["enabled"]:
        wandb_logger = WandbLogger(wandb_config, "baseline")
        # Log configurations
        wandb_logger.update_config({
            "data_config": data_config,
            "model_config": model_config,
            "training_config": training_config
        })
    else:
        wandb_logger = None
    
    # Initialize trainer
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
        wandb_logger=wandb_logger
    )
    
    # Train model
    print("\nStarting training...")
    model, history = trainer.train()
    
    # Save training history
    history_path = os.path.join(config_manager.get_model_dir("baseline"), "training_history.json")
    save_json(history, history_path)
    print(f"\nSaved training history to {history_path}")
    
    # Plot training history
    plot_path = os.path.join(config_manager.get_model_dir("baseline"), "training_curves.png")
    plot_training_history(history, model_name="Baseline", save_path=plot_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_bleu = calculate_bleu(model, test_loader, vocab, device, 
                              max_samples=training_config.get("max_bleu_samples"))
    
    print("\nTest Set BLEU Scores:")
    print(f"BLEU-1: {test_bleu['bleu1']:.2f}")
    print(f"BLEU-2: {test_bleu['bleu2']:.2f}")
    print(f"BLEU-3: {test_bleu['bleu3']:.2f}")
    print(f"BLEU-4: {test_bleu['bleu4']:.2f}")
    
    # Save test results
    test_results = {
        "model": "baseline",
        "test_bleu": test_bleu,
        "model_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "timestamp": datetime.now().isoformat()
    }
    
    results_path = os.path.join(config_manager.get_model_dir("baseline"), "test_results.json")
    save_json(test_results, results_path)
    print(f"\nSaved test results to {results_path}")
    
    # Visualize sample captions
    print("\nGenerating sample captions...")
    viz_dir = os.path.join(config_manager.get_model_dir("baseline"), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    visualize_sample_captions(model, test_dataset, vocab, device, num_samples=5, save_dir=viz_dir)
    
    # Finish wandb run
    if wandb_logger:
        # Log final test results
        wandb_logger.log_metrics(test_results, step=trainer.global_step)
        wandb_logger.finish()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()