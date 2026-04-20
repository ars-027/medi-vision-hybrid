import os
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRanged, CropForegroundd, Spacingd, Orientationd, 
    RandCropByPosNegLabeld, EnsureTyped
)
from monai.networks.nets import UNETR  
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ==========================================
# 1. CONFIGURATION & REAL DATA PIPELINE
# ==========================================
# We use 96x96x96 patches to fit the RTX 3050 VRAM while training on large real CTs
spatial_size = (96, 96, 96) 
batch_size = 2 
data_dir = "./msd_data"
os.makedirs(data_dir, exist_ok=True)

# The Clinical Transformation Pipeline for CT Scans
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(   
        keys=["image"], a_min=-57, a_max=164, # Windowing specifically for soft tissue
        b_min=0.0, b_max=1.0, clip=True,
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=spatial_size,
        pos=1, neg=1, num_samples=2, # Extracts 2 random patches per volume per epoch
        image_key="image", image_threshold=0,
    ),
    EnsureTyped(keys=["image", "label"])
])

print("Initializing Medi-Vision Hybrid: Phase 2...")
print("Downloading/Locating Medical Segmentation Decathlon Data (Task09_Spleen)...")

# DecathlonDataset handles downloading, extracting, and caching
train_ds = DecathlonDataset(
    root_dir=data_dir,
    task="Task09_Spleen",
    transform=train_transforms,
    section="training",
    download=True,  # Will only download the 1.5GB file once
    cache_rate=0.0, # Set to 0.0 to save RAM during initial local testing
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# ==========================================
# 2. MODEL, LOSS, & OPTIMIZER
# ==========================================
model = UNETR(
    in_channels=1,
    out_channels=2,      # 2 classes: Background (0) and Spleen (1)
    img_size=spatial_size,
    feature_size=16,
    hidden_size=768,     
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron", 
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to("cuda")

# CrossEntropy + Dice is the clinical standard for real anatomical data
loss_function = DiceLoss(to_onehot_y=True, softmax=True) 
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ==========================================
# 3. THE TRAINING & VALIDATION LOOP
# ==========================================
max_epochs = 50 # Increased for real data convergence
val_interval = 2 
best_metric = -1
best_metric_epoch = -1

for epoch in range(max_epochs):
    print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        # Real datasets load as dictionaries, so we use string keys instead of index [0]
        inputs, labels = batch_data["image"].to("cuda"), batch_data["label"].to("cuda")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= step
    print(f"Average Training Loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in train_loader: 
                val_inputs, val_labels = val_data["image"].to("cuda"), val_data["label"].to("cuda")
                
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, 
                    roi_size=spatial_size, 
                    sw_batch_size=batch_size, 
                    predictor=model
                )
                
                # Convert outputs to discrete predictions for metric calculation
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "medivision_hybrid_clinical_best.pth")
                print(f"New Best Dice Score: {best_metric:.4f}! Weights saved.")
            else:
                print(f"Current Dice Score: {metric:.4f} (Best: {best_metric:.4f})")

print(f"\nTraining Complete. Best Metric: {best_metric:.4f} at Epoch {best_metric_epoch}")