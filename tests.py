# test_imports.py

"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    # Utils
    from src.utils.constants import SEED
    from src.utils.config import load_config
    from src.utils.manager import ConfigManager
    from src.utils.io import save_pickle, load_pickle
    from src.utils.logger import VerboseLogger
    from src.utils.wanlog import WandbLogger
    print("Utils imports successful")
    
    # Preprocessing
    from src.preprocessing.vocabulary import Vocabulary
    from src.preprocessing.transforms import get_transforms
    from src.preprocessing.dataset import FlickrDataset
    print("Preprocessing imports successful")
    
    # Models
    from src.models.encoders import EncoderCNN
    from src.models.baseline import BaselineCaptionModel
    from src.models.attention import AttentionCaptionModel
    print("Models imports successful")
    
    # Training
    from src.training.metrics import calculate_bleu
    from src.training.trainer import Trainer
    print("Training imports successful")
    
    # Visualization
    from src.visualization.attention import visualize_attention
    from src.visualization.captioning import plot_training_history
    print("Visualization imports successful")
    
    # Comparison
    from src.comparison.evaluator import ModelEvaluator
    print("Comparison imports successful")
    
    print("\nAll imports successful! The project structure is set up correctly")

if __name__ == "__main__":
    test_imports()