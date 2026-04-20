import torch
from torch.utils.data import Dataset
from monai.data import DataLoader, create_test_image_3d
from monai.networks.nets import UNETR  
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ==========================================
# 1. CONFIGURATION & DATA PIPELINE
# ==========================================
spatial_size = (128, 128, 128)
batch_size = 2 # Optimized for RTX 3050

class SyntheticCTDataset(Dataset):
    def __init__(self, data_count, spatial_shape):
        self.data_count = data_count
        self.spatial_shape = spatial_shape
        
    def __len__(self):
        return self.data_count
        
    def __getitem__(self, idx):
        # Generates a random 3D volume with a synthetic "tumor"
        img, label = create_test_image_3d(
            self.spatial_shape[0], 
            self.spatial_shape[1], 
            self.spatial_shape[2], 
            num_objs=1,          
            rad_max=20,          
            num_seg_classes=1,   
            channel_dim=0        
        )
        return torch.tensor(img).float(), torch.tensor(label).float()

print("Initializing Medi-Vision Hybrid Pipeline...")
# Create 10 mock samples (Use a small number for local testing)
val_ds = SyntheticCTDataset(data_count=10, spatial_shape=spatial_size)
train_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# ==========================================
# 2. MODEL, LOSS, & OPTIMIZER
# ==========================================
model = UNETR(
    in_channels=1,
    out_channels=1,      
    img_size=spatial_size,
    feature_size=16,
    hidden_size=768,     
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron", # Updated API parameter
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to("cuda")

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ==========================================
# 3. THE TRAINING & VALIDATION LOOP
# ==========================================
max_epochs = 10
val_interval = 2 
best_metric = -1
best_metric_epoch = -1

for epoch in range(max_epochs):
    print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
    
    # --- TRAINING PASS ---
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to("cuda"), batch_data[1].to("cuda")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= step
    print(f"Average Training Loss: {epoch_loss:.4f}")
    
    # --- VALIDATION PASS ---
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in train_loader: 
                val_inputs, val_labels = val_data[0].to("cuda"), val_data[1].to("cuda")
                
                # Sliding window inference for large 3D volumes
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, 
                    roi_size=spatial_size, 
                    sw_batch_size=batch_size, 
                    predictor=model
                )
                
                # Thresholding probabilities to create a binary mask
                val_outputs = (val_outputs.sigmoid() > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            # Calculate final Dice score
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # Save the best weights
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "medivision_hybrid_best_model.pth")
                print(f"New Best Dice Score: {best_metric:.4f}! Weights saved.")
            else:
                print(f"Current Dice Score: {metric:.4f} (Best: {best_metric:.4f})")

print(f"\nTraining Complete. Best Metric: {best_metric:.4f} at Epoch {best_metric_epoch}")