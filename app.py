import os
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
import io
import json
import warnings
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import google.generativeai as genai

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    Spacingd, Orientationd, EnsureTyped, KeepLargestConnectedComponent
)
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. INITIALIZE HYBRID AI ENGINE & APIs
# ==========================================
# 🚨 PASTE YOUR GOOGLE AI STUDIO API KEY BELOW 🚨
genai.configure(api_key="AIzaSyA9rpbHPVKw7iDZW0kz4kZe5TqfcUATPIg")

generation_config = {"response_mime_type": "application/json"}
llm_model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

app = FastAPI(title="Medi-Vision Enterprise Engine", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 2. LOCAL EHR DATABASE INGESTION
# ==========================================
print("Connecting to Secure Local EHR Data Store...")

@app.get("/patients")
def get_patients():
    """Dynamically queries the local CSV data store and serves it to the UI"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "patients.csv")
        df_patients = pd.read_csv(db_path)
        records_str = df_patients.fillna("").to_json(orient="records")
        return JSONResponse(content={"patients": json.loads(records_str)})
    except Exception as e:
        print(f"Database Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Local Database Offline"})

# ==========================================
# 3. INITIALIZE DETERMINISTIC AI (UNETR)
# ==========================================
spatial_size = (96, 96, 96)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing UNETR Segmentation Core...")
unetr_model = UNETR(
    in_channels=1, out_channels=2, img_size=spatial_size,
    feature_size=16, hidden_size=768, mlp_dim=3072,
    num_heads=12, proj_type="perceptron", norm_name="instance",
    res_block=True, dropout_rate=0.0,
).to(device)

try:
    unetr_model.load_state_dict(torch.load("medivision_hybrid_clinical_best.pth", weights_only=True))
    unetr_model.eval()
except Exception as e:
    print("WARNING: PyTorch weights not found. UNETR branch will fail if triggered.")

inference_transforms = Compose([
    LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image"])
])

# ==========================================
# 4. MULTI-MODAL ROUTING ENGINE
# ==========================================
@app.post("/predict")
async def predict_scan(file: UploadFile = File(...), patient_context: str = Form(default="None provided")):
    filename = file.filename.lower()
    
    # ---------------------------------------------------------
    # BRANCH A: 2D RAW IMAGE (Vision Path)
    # ---------------------------------------------------------
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        print(f"Routing {filename} with context: {patient_context}")
        image_data = await file.read()
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        prompt = f"""
        You are an expert radiologist. Analyze this 2D medical image.
        CLINICAL CONTEXT: {patient_context}
        Take this patient history into account when analyzing the image. 
        
        Return a strict JSON object with these exact keys:
        "primary_finding": A one-sentence bold diagnosis.
        "detailed_analysis": A short paragraph explaining tissue state, referencing the clinical context if relevant.
        "risk_level": Must be exactly "Low", "Medium", or "High".
        "recommendation": What the next clinical step should be.
        "confidence_score": A percentage (e.g., "94%") representing how certain you are based on image clarity.
        """
        try:
            llm_response = llm_model.generate_content([prompt, pil_img])
            diagnostic_data = json.loads(llm_response.text)
        except Exception as e:
            diagnostic_data = {
                "primary_finding": "API Connection Error",
                "detailed_analysis": str(e),
                "risk_level": "High",
                "recommendation": "Check API connection.",
                "confidence_score": "0%"
            }
            
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return JSONResponse(content={
            "image_base64": encoded_image,
            "volume_cc": "N/A (2D Image)",
            "diagnosis": diagnostic_data
        })

    # ---------------------------------------------------------
    # BRANCH B: 3D VOLUME (UNETR Path)
    # ---------------------------------------------------------
    elif filename.endswith(('.nii', '.nii.gz')):
        print(f"Routing {filename} to UNETR Pipeline with context: {patient_context}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        try:
            ds = Dataset(data=[{"image": temp_path}], transform=inference_transforms)
            loader = DataLoader(ds, batch_size=1)
            case = next(iter(loader))
            inputs = case["image"].to(device)

            with torch.no_grad():
                torch.cuda.empty_cache() 
                outputs = sliding_window_inference(inputs, spatial_size, 1, unetr_model)
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                post_process = KeepLargestConnectedComponent(applied_labels=[1])
                outputs_clean = post_process(outputs[0])
                
                outputs_np = outputs_clean.cpu().numpy()[0]
                inputs_np = inputs.cpu().numpy()[0, 0]

            voxel_count = np.sum(outputs_np)
            total_volume_cc = round((voxel_count * 4.5) / 1000, 2)

            spleen_areas = outputs_np.sum(axis=(0, 1))
            slice_idx = spleen_areas.argmax() if spleen_areas.max() > 0 else inputs_np.shape[2] // 2
            ct_slice = inputs_np[:, :, slice_idx]
            mask_slice = outputs_np[:, :, slice_idx]

            fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0b1120')
            axes[0].set_facecolor('#0b1120'); axes[0].imshow(ct_slice, cmap="gray"); axes[0].axis("off")
            axes[1].set_facecolor('#0b1120'); axes[1].imshow(ct_slice, cmap="gray")
            cmap_overlay = mcolors.ListedColormap(['none', '#ef4444'])
            axes[1].imshow(mask_slice, cmap=cmap_overlay, alpha=0.55); axes[1].axis("off")
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            buf.seek(0)
            pil_img = Image.open(buf)
            
            prompt = f"""
            You are an expert radiologist. Analyze this abdominal CT scan slice. 
            The red highlight represents the patient's spleen, mathematically measured at {total_volume_cc} cc. 
            Normal spleen volume is 150-250 cc. Splenomegaly is >314 cc.
            
            CLINICAL CONTEXT: {patient_context}
            Take this patient history into account.
            
            Return a strict JSON object with keys: "primary_finding", "detailed_analysis", "risk_level" (Low/Medium/High), "recommendation", and "confidence_score" (e.g. "98%").
            """
            
            try:
                llm_response = llm_model.generate_content([prompt, pil_img])
                diagnostic_data = json.loads(llm_response.text)
            except Exception as e:
                diagnostic_data = {
                    "primary_finding": "API Error", "detailed_analysis": str(e),
                    "risk_level": "High", "recommendation": "Check API", "confidence_score": "0%"
                }

            buf.seek(0)
            encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')

            return JSONResponse(content={
                "image_base64": encoded_image,
                "volume_cc": total_volume_cc,
                "diagnosis": diagnostic_data
            })
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
            
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported format."})