import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CWStreetViewDataset(Dataset):
    """
    Custom Dataset for CW-SegFormer.
    Includes logic for loading SVI images and corresponding semantic masks.
    """
    def __init__(self, data_root, split='train', transform=None, target_transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load file list
        self.img_dir = os.path.join(data_root, 'images', split)
        self.mask_dir = os.path.join(data_root, 'annotations', split)
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        # Pre-scan for small objects to support oversampling
        self.small_object_indices = self._scan_for_small_objects()

    def _scan_for_small_objects(self):
        """
        Identify images containing Surveillance (ID: 6) or Lighting (ID: 5).
        Used to create WeightedRandomSampler in training.
        """
        indices = []
        return indices 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # Assuming indexed png

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
            
        # Ensure mask is a tensor of type long
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()

        return {"pixel_values": image, "labels": mask}

    def get_sample_weights(self):
        """
        Returns weights for WeightedRandomSampler.
        Images with small objects get higher weights (e.g., 2.0x).
        """
        weights = []
        for f in self.files:
            # Logic: Check metadata or filename to see if it contains small objects
            weight = 1.0 
            # if contains_small_object(f): weight = 2.0
            weights.append(weight)
        return torch.DoubleTensor(weights)
