import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    Spacingd, Orientationd, EnsureTyped, KeepLargestConnectedComponent, AsDiscrete
)
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader

print("Initiating UNETR Post-Processing Ablation Study...")

# 1. Setup Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNETR(
    in_channels=1, out_channels=2, img_size=(96, 96, 96), feature_size=16, 
    hidden_size=768, mlp_dim=3072, num_heads=12, proj_type="perceptron", 
    norm_name="instance", res_block=True, dropout_rate=0.0
).to(device)

model.load_state_dict(torch.load("medivision_hybrid_clinical_best.pth", weights_only=True))
model.eval()

# Metrics: One for Raw (Base ViT), One for Cleaned (Post-Processed)
dice_raw = DiceMetric(include_background=False, reduction="mean")
dice_clean = DiceMetric(include_background=False, reduction="mean")

post_raw = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_clean = Compose([AsDiscrete(argmax=True, to_onehot=2), KeepLargestConnectedComponent(applied_labels=[1])])
post_label = Compose([AsDiscrete(to_onehot=2)])

# 2. Load Validation Data (Using just 5 for a fast ablation test)
data_dir = "./msd_data/Task09_Spleen"
val_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))[-5:]
val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))[-5:]

val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"])
])
val_loader = DataLoader(Dataset(data=val_files, transform=val_transforms), batch_size=1, num_workers=0)

# 3. Execution Loop
results = []
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        file_name = os.path.basename(val_files[i]["image"])
        print(f"Analyzing {file_name}...")

        # Raw Inference
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
        
        # Branch A: Raw ViT Output
        raw_list = [post_raw(i) for i in val_outputs]
        # Branch B: Post-Processed Output
        clean_list = [post_clean(i) for i in val_outputs]
        label_list = [post_label(i) for i in val_labels]

        dice_raw(y_pred=raw_list, y=label_list)
        dice_clean(y_pred=clean_list, y=label_list)

        results.append({
            "Scan": file_name,
            "Raw UNETR": dice_raw.aggregate().item(),
            "UNETR + Post-Processing": dice_clean.aggregate().item()
        })
        dice_raw.reset(); dice_clean.reset()

# 4. Generate Comparative Plot
df = pd.DataFrame(results).melt(id_vars="Scan", var_name="Pipeline Stage", value_name="Dice Score")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Grouped bar chart to show the exact improvement per scan
sns.barplot(data=df, x="Scan", y="Dice Score", hue="Pipeline Stage", palette=["#64748b", "#0ea5e9"])
plt.title("Ablation Study: Impact of Morphological Post-Processing", fontsize=16, pad=15, fontweight='bold')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=15)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("ablation_study_results.png", dpi=300)
print("\nAblation complete! Saved 'ablation_study_results.png'.")