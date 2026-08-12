from tifffile import logger
import torch

class Config:
    def __init__(self):
        # ============================
        # Device & Seed
        # ============================
        self.gpu_name = 'cuda:0'
        self.device = torch.device(self.gpu_name if torch.cuda.is_available() else "cpu")
        self.seed = 3407  # Can select: 42, 3407, 2025
        self.use_amp = True  # Whether to use automatic mixed precision training        
        self.use_wandb = True  # Whether to use Weights & Biases for experiment tracking
        self.offline_preprocessing = True  # Whether to perform offline preprocessing (True → preprocess and save, False → online preprocessing)
        
        # ============================
        # Dataset Paths & Modalities
        # ============================
        # BraTS2026GoAT
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        self.modality_separator = "-"
        self.image_suffix = ".nii.gz"
        self.et_label = 3
        self.num_classes = 3
        
        # ============================
        # Dataset & Preprocessing
        # ============================
        self.overlap = 0.125  # Sliding window overlap ratio
        self.num_workers = 8  # Number of DataLoader parallel threads

        # ============================
        # Patch & Input Settings
        # ============================
        self.patch_size = [128, 128, 128]  # Options: [64,64,64], [96,96,96], [128,128,128]
        self.inference_patch_size = [128, 128, 128]  # Patch size during inference            
        
# Global singleton
config = Config()
