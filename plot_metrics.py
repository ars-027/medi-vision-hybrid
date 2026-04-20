import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Generating clinical validation plots...")

# 1. Load the data generated from our evaluation script
try:
    df = pd.read_csv("clinical_validation_report.csv")
except FileNotFoundError:
    print("Error: Could not find 'clinical_validation_report.csv'. Make sure you ran the evaluation script first!")
    exit()

# 2. Set the aesthetic theme for professional presentation slides
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3. Plot 1: Dice Score (Accuracy)
# Using your UI's primary blue color
sns.boxplot(y=df["Dice_Score"], ax=axes[0], color="#0ea5e9", width=0.4, linewidth=2)
sns.stripplot(y=df["Dice_Score"], ax=axes[0], color=".25", size=6, alpha=0.6) # Shows individual scan dots
axes[0].set_title("Volumetric Accuracy (Dice Score)", fontsize=14, pad=15, fontweight='bold')
axes[0].set_ylabel("Dice Score (1.0 = Perfect Overlap)", fontsize=12)
axes[0].set_ylim(0, 1.05) 

# 4. Plot 2: Hausdorff Distance (Boundary Error)
# Using your UI's accent red color
sns.boxplot(y=df["Hausdorff_95_mm"], ax=axes[1], color="#ef4444", width=0.4, linewidth=2)
sns.stripplot(y=df["Hausdorff_95_mm"], ax=axes[1], color=".25", size=6, alpha=0.6)
axes[1].set_title("Boundary Error (Hausdorff Distance 95%)", fontsize=14, pad=15, fontweight='bold')
axes[1].set_ylabel("Distance in mm (Lower is Better)", fontsize=12)

# 5. Final Formatting & Export
plt.suptitle("Medi-Vision Hybrid: Phase 1 Clinical Validation Benchmarks", fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()

output_filename = "clinical_metrics_boxplot.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Success! High-resolution plot saved as '{output_filename}' ready for your slides.")