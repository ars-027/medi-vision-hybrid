# medi-vision-hybrid
A Cross-Attentional Platform for Explainable Medical Imaging.

# 🏥 Medi-Vision Hybrid: A Cross-Attentional Platform for Explainable Medical Imaging

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-MONAI-red)
![JavaScript](https://img.shields.io/badge/Frontend-Vanilla_JS-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Project Overview
Current medical AI systems frequently operate as isolated "black boxes," offering high-accuracy visual recognition but lacking the clinical context required for confident physician adoption. 

**Medi-Vision Hybrid** is a cross-attentional enterprise platform designed to automate radiological triage. It bridges the gap between deterministic deep learning and generative reasoning by implementing a dual-pipeline routing engine. By utilizing **Retrieval-Augmented Generation (RAG)**, the platform fuses raw imaging data with offline patient history, ensuring that the AI evaluates scans contextually, drastically reducing false positives.

## ✨ Key Features
* **Multi-Modal Routing Engine:** Automatically detects and routes 3D volumetric data (NIfTI) and 2D planar radiographs (PNG/JPG) to specialized processing pipelines.
* **3D Deterministic Pipeline:** Utilizes **UNETR (UNet Transformers)** and the MONAI framework to perform precise spatial segmentation, calculating absolute organ volumes (e.g., Spleen volume in cubic centimeters).
* **Context-Aware Generative RAG:** Integrates Large Multimodal Models (LMMs) with a localized offline EHR database (`patients.csv`) to generate explainable, context-grounded diagnostic reports.
* **AI Safety Gatekeeper:** A hardcoded validation protocol that intercepts and blocks non-clinical images, ensuring a zero-hallucination environment.
* **Enterprise Compliance:** Features immutable session Audit Logging with serialized Traceability IDs and automated one-page clinical PDF handover generation (`html2pdf.js`).

## 🛠️ Technology Stack
* **Backend:** Python, FastAPI, Uvicorn, Pandas
* **Deep Learning:** PyTorch, MONAI (Medical Open Network for AI), NumPy
* **Generative AI:** Google GenAI SDK (Gemini 2.5 Flash API)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, html2pdf.js

## 🚀 Installation and Setup

### Prerequisites
* Python 3.10 or higher
* Node.js (Optional, for frontend package management if expanding)
* A valid Google Gemini API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/medi-vision-hybrid.git](https://github.com/YOUR-USERNAME/medi-vision-hybrid.git)
cd medi-vision-hybrid
```

### 2. Set Up the Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API key:
```env
GEMINI_API_KEY=your_api_key_here
```
*(Note: Do not upload your `.env` file or API keys to GitHub!)*

### 5. Run the Backend Server
```bash
uvicorn app.main:app --reload
```
The FastAPI backend will now be running on `http://localhost:8000`.

### 6. Launch the Frontend
Simply open the `index.html` file in your preferred web browser, or serve it using a lightweight local server (e.g., `python -m http.server 3000`).

## 🔮 Future Scope
* **DICOM Integration:** Upgrading from local NIfTI drag-and-drop to a direct PyDICOM network node to fetch scans from hospital PACS.
* **Air-Gapped LMMs:** Replacing the cloud-based Gemini API with fully on-premise, open-source vision models (such as **LLaVA**) to ensure 100% HIPAA compliance and data privacy.

