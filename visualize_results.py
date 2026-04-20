from monai.transforms import KeepLargestConnectedComponent
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    CropForegroundd, Spacingd, Orientationd, EnsureTyped
)
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference


# ==========================================
# 1. SETUP DATA FOR INFERENCE
# ==========================================
spatial_size = (96, 96, 96)
data_dir = "./msd_data"

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"])
])

val_ds = DecathlonDataset(root_dir=data_dir, task="Task09_Spleen", transform=val_transforms, section="validation", download=False)
val_loader = DataLoader(val_ds, batch_size=1)

# ==========================================
# 2. LOAD THE SAVED WEIGHTS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNETR(
    in_channels=1, out_channels=2, img_size=spatial_size,
    feature_size=16, hidden_size=768, mlp_dim=3072,
    num_heads=12, proj_type="perceptron", norm_name="instance",
    res_block=True, dropout_rate=0.0,
).to(device)

# Load weights securely
model.load_state_dict(torch.load("medivision_hybrid_clinical_best.pth", weights_only=True))
model.eval()

# ==========================================
# 3. GENERATE PREDICTION & POST-PROCESSING
# ==========================================
print("Generating visualization for a test scan...")
with torch.no_grad():
    case = next(iter(val_loader)) 
    val_inputs = case["image"].to(device)
    
    # 1. Raw Inference
    val_outputs = sliding_window_inference(val_inputs, spatial_size, 4, model)
    
    # Keep the channel dimension for the transform: Shape becomes [Batch, Channel, H, W, D]
    val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True) 
    
    # 2. Post-Processing: Erase false-positive "islands"
    # This specifically looks for the label '1' (Spleen) and keeps only the largest blob
    post_process = KeepLargestConnectedComponent(applied_labels=[1])
    val_outputs_clean = post_process(val_outputs[0]) # Apply to the single item in the batch
    
    # 3. Format for Matplotlib
    val_outputs_np = val_outputs_clean.detach().cpu()[0].numpy() 
    val_labels_np = case["label"][0, 0].cpu().numpy() 
    val_image_np = val_inputs.cpu()[0, 0].numpy()

# ==========================================
# 4. DYNAMIC SLICE SELECTION & PLOTTING
# ==========================================
# Automatically find the slice with the maximum spleen area
spleen_areas = val_labels_np.sum(axis=(0, 1))
slice_idx = spleen_areas.argmax()

if spleen_areas[slice_idx] == 0:
    slice_idx = val_image_np.shape[2] // 2 # Fallback to the physical center

print(f"Plotting Slice {slice_idx} (Maximum Spleen Area detected)")

plt.figure("Medi-Vision Hybrid Inference", figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original CT Slice")
plt.imshow(val_image_np[:, :, slice_idx], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth (Radiologist)")
plt.imshow(val_labels_np[:, :, slice_idx], cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("AI Prediction (UNETR)")
plt.imshow(val_outputs_np[:, :, slice_idx], cmap="jet")
plt.axis("off")

plt.tight_layout()
plt.show()