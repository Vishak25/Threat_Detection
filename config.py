import os
import sys
from pathlib import Path

class Config:
    def __init__(self):
        self.IS_COLAB = 'google.colab' in sys.modules
        self.IS_LINUX = sys.platform.startswith("linux")
        self.IS_MAC = sys.platform == "darwin"
        
        # Hyperparameters
        self.BATCH_SIZE = 32
        self.EPOCHS = 20
        self.LEARNING_RATE = 1e-4
        
        # MIL Parameters
        self.MIL_K = 32 # Top-k instances
        self.SELECTED_LABELS = ['Arrest', 'Fighting'] # Specific labels to use
        self.MAX_LABELS = None # Deprecated in favor of SELECTED_LABELS
        
        # Paths
        # We assume the script/notebook has already set the working directory correctly
        self.ROOT_DIR = Path.cwd()
        self.DATA_DIR = self.ROOT_DIR / "data/DCSASS Dataset"
        self.CHECKPOINT_DIR = self.ROOT_DIR / "checkpoints"
            
    def display(self):
        print(f"Environment: {'Colab' if self.IS_COLAB else 'Local'}")
        print(f"Root Dir: {self.ROOT_DIR}")
        print(f"Data Dir: {self.DATA_DIR}")
        print(f"Batch Size: {self.BATCH_SIZE}")

config = Config()
