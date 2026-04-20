import os
import time
import torch
import pandas as pd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    Spacingd, Orientationd, EnsureTyped, KeepLargestConnectedComponent,
    AsDiscrete
)
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import Dataset, DataLoader

print("Initializing Phase 1: Clinical Validation Protocol...")

# ==========================================
# 1. SETUP ENGINE & METRICS
# ==========================================
spatial_size = (96, 96, 96)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1, out_channels=2, img_size=spatial_size,
    feature_size=16, hidden_size=768, mlp_dim=3072,
    num_heads=12, proj_type="perceptron", norm_name="instance",
    res_block=True, dropout_rate=0.0,
).to(device)

print("Loading optimized weights...")
model.load_state_dict(torch.load("medivision_hybrid_clinical_best.pth", weights_only=True))
model.eval()

# MONAI metrics require specific formatting (One-Hot Encoding)
dice_metric = DiceMetric(include_background=False, reduction="mean")
hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2), KeepLargestConnectedComponent(applied_labels=[1])])
post_label = Compose([AsDiscrete(to_onehot=2)])

# ==========================================
# 2. LOAD VALIDATION DATA
# ==========================================
# IMPORTANT: Update these paths to match your local Decathlon dataset structure
data_dir = "./msd_data/Task09_Spleen"
import glob
val_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))[-10:] # Using last 10 as validation
val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))[-10:]

val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"])
])

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# ==========================================
# 3. RUN EVALUATION LOOP
# ==========================================
results = []
print(f"Beginning evaluation on {len(val_files)} clinical volumes...\n")

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        start_time = time.time()
        
        val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        file_name = os.path.basename(val_files[i]["image"])
        print(f"Processing {file_name}...")

        # 1. Inference
        val_outputs = sliding_window_inference(val_inputs, spatial_size, 4, model)
        
        # 2. Format tensors for metrics
        val_outputs_list = [post_pred(i) for i in val_outputs]
        val_labels_list = [post_label(i) for i in val_labels]

        # 3. Compute Metrics
        dice_metric(y_pred=val_outputs_list, y=val_labels_list)
        hd_metric(y_pred=val_outputs_list, y=val_labels_list)
        
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)

        # 4. Extract scores for this specific volume
        current_dice = dice_metric.aggregate().item()
        current_hd = hd_metric.aggregate().item()
        
        # Reset metric for the next volume to avoid averaging bugs
        dice_metric.reset()
        hd_metric.reset()

        results.append({
            "Scan_ID": file_name,
            "Dice_Score": round(current_dice, 4),
            "Hausdorff_95_mm": round(current_hd, 4),
            "Inference_Time_sec": inference_time
        })

# ==========================================
# 4. EXPORT REPORT
# ==========================================
df = pd.DataFrame(results)
df.to_csv("clinical_validation_report.csv", index=False)

print("\n" + "="*40)
print("EVALUATION COMPLETE")
print("="*40)
print(f"Mean Dice Score:       {df['Dice_Score'].mean():.4f}")
print(f"Mean Hausdorff (95%):  {df['Hausdorff_95_mm'].mean():.4f} mm")
print(f"Mean Inference Time:   {df['Inference_Time_sec'].mean():.2f} sec")
print("\nDetailed report saved to: clinical_validation_report.csv")