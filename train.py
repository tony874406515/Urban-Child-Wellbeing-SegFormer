import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm

# Import project-specific modules
from dataset import CWStreetViewDataset
from utils import CLASSES, CLASS_WEIGHTS, compute_metrics

def train_cw_segformer():
    # 1. Load Configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device(config['training_parameters']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Initializing {config['project_name']} on {device}...")

    # 2. Data Preparation
    # Define transforms (Standardization as mentioned in Section 3.1)
    train_transform = transforms.Compose([
        transforms.Resize(config['model_architecture']['image_size']),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset initialization
    train_dataset = CWStreetViewDataset(
        data_root=config['paths']['data_root'], 
        split='train', 
        transform=train_transform
    )
    
    # Strategy: Oversampling for small objects (Manuscript Table 4)
    # We use a WeightedRandomSampler to increase the frequency of images with small targets
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training_parameters']['batch_size'], 
        sampler=sampler, # Apply oversampling strategy
        num_workers=4
    )

    # 3. Model Initialization (CW-SegFormer)
    # Loading the base SegFormer-B4 and adjusting the head for 10 classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        config['model_architecture']['base_model'],
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=config['training_parameters']['learning_rate'])
    
    num_training_steps = len(train_loader) * config['training_parameters']['max_epochs']
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config['training_parameters']['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # 5. Loss Function with Class Weights
    # Higher weights for "Surveillance" and "Lighting" to improve Recall
    class_weights_tensor = torch.tensor(CLASS_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 6. Training Loop
    model.train()
    print("Starting training...")
    
    # Simplified loop for demonstration
    for epoch in range(5): # Demonstrate a few epochs
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            images = batch['pixel_values'].to(device)
            masks = batch['labels'].to(device)

            optimizer.zero_grad()
            
            outputs = model(pixel_values=images)
            # SegFormer outputs logits are 1/4th resolution, need upsampling
            logits = nn.functional.interpolate(
                outputs.logits, 
                size=images.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss / len(train_loader)}")
        
        # Validation and saving logic would go here...
        
    print("Training complete. Model saved to", config['paths']['output_dir'])
    torch.save(model.state_dict(), os.path.join(config['paths']['output_dir'], "cw_segformer_best.pth"))

if __name__ == "__main__":
    train_cw_segformer()
