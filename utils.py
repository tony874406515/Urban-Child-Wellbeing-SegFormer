import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------------
# Label Mapping (Consistent with Manuscript Table 1)
# -------------------------------------------------------------------------
CLASSES = [
    "Greenery", "Building_facade", "Pedestrian_cycleway", "Motorway",
    "Street_furniture", "Lighting_facilities", "Surveillance_equipment",
    "Traffic_signs", "Pedestrians_bicycles", "Windows"
]

# Mapping from internal ID (0-9) to visualization color (RGB)
PALETTE = [
    [0, 255, 0],     # Greenery
    [128, 128, 128], # Building_facade
    [244, 35, 232],  # Pedestrian_cycleway
    [0, 0, 142],     # Motorway
    [190, 153, 153], # Street_furniture
    [250, 170, 30],  # Lighting_facilities (Important for Safety)
    [220, 220, 0],   # Surveillance_equipment (Important for Safety)
    [220, 20, 60],   # Traffic_signs
    [255, 0, 0],     # Pedestrians_bicycles
    [70, 70, 70]     # Windows
]

# Specific weights for CrossEntropyLoss to handle class imbalance
# Higher weights for small objects (Surveillance, Lighting, Signs)
# These values are derived from pixel frequency analysis in the study area
CLASS_WEIGHTS = [1.0, 1.0, 1.2, 1.0, 1.5, 3.0, 3.0, 2.5, 2.0, 1.5]

def compute_metrics(pred_mask, true_mask, num_classes=10):
    """
    Computes mIoU, Pixel Accuracy, and Class-wise Recall.
    Used for Table 4 validation in the manuscript.
    """
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    
    # Filter out ignore indices (if any)
    valid_indices = true_mask != 255
    pred_mask = pred_mask[valid_indices]
    true_mask = true_mask[valid_indices]

    hist = confusion_matrix(true_mask, pred_mask, labels=range(num_classes))
    
    # Pixel Accuracy
    acc = np.diag(hist).sum() / hist.sum()
    
    # IoU
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    mean_iou = np.nanmean(iu)
    
    # Recall (Class-wise Accuracy)
    recall = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
    
    return {
        "Pixel Accuracy": acc,
        "mIoU": mean_iou,
        "Class Recall": recall
    }
