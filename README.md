# 🦴 Knee OA Analyzer
### AI-Assisted Severity Classification of Knee Osteoarthritis

An end-to-end clinical decision support system that automatically grades knee osteoarthritis severity from X-ray images using a deep learning ensemble model, with Grad-CAM explainability and automated PDF report generation.

---

## 📱 Features

- **KL Grade Prediction** — Classifies knee X-rays into Kellgren-Lawrence Grade 0–4
- **Ensemble Deep Learning** — EfficientNet-B5 + EfficientNet-V2-S + DenseNet-201
- **Grad-CAM Visualization** — Highlights regions the model focused on
- **PDF Report Generation** — Professional clinical report with findings
- **Patient History** — Stores and manages all past assessments
- **Doctor Login** — Secure access for authorized clinicians
- **Camera + Gallery** — Capture or upload X-ray images directly

---

## 🏗️ Architecture
Flutter Mobile App (Frontend)
↕ REST API
FastAPI Backend (Python)
↕
PyTorch Ensemble Model
(EfficientNet-B5 + EfficientNet-V2-S + DenseNet-201)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile Frontend | Flutter (Dart) |
| Backend API | Python FastAPI |
| Deep Learning | PyTorch |
| Model Architectures | EfficientNet-B5, EfficientNet-V2-S, DenseNet-201 |
| Explainability | Grad-CAM |
| PDF Generation | ReportLab |
| Image Processing | OpenCV, Pillow |



## 📁 Project Structure
```
knee_oa_app/
├── lib/
│   └── main.dart              # Flutter app — Login, Home, History screens
├── backend/
│   ├── main.py                # FastAPI server — prediction, history, reports
│   ├── report.py              # PDF report generation
│   ├── efficientnet_b5_best.pt      # Trained model weights
│   ├── efficientnet_v2_s_best.pt    # Trained model weights
│   └── densenet_201_best.pt         # Trained model weights
└── pubspec.yaml
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- Flutter 3.x
- PyTorch

### Backend Setup

```bash
cd knee_oa_app/backend
pip install fastapi uvicorn torch torchvision pillow python-multipart opencv-python reportlab
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd knee_oa_app
flutter pub get
flutter run -d chrome
```

> **Note:** Add your trained `.pt` model files to the `backend/` directory before running.

---

## 🧠 Model Details

The system uses a **weighted probability averaging ensemble** of three CNN architectures:

| Model | Backbone | Strength |
|-------|----------|----------|
| EfficientNet-B5 | EfficientNet family | High accuracy, efficient |
| EfficientNet-V2-S | EfficientNet V2 family | Fast training, compact |
| DenseNet-201 | DenseNet family | Feature reuse, fine-grained |

**Dataset:** Knee Osteoarthritis Dataset with Severity Grading (Kaggle)  
**Classes:** KL Grade 0 (Normal) → Grade 4 (Severe)  
**Preprocessing:** CLAHE, resize to 224×224, ImageNet normalization

---

## 📊 KL Grading Scale

| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | Normal | No OA features |
| 1 | Doubtful | Possible osteophyte |
| 2 | Mild | Definite osteophytes, possible JSN |
| 3 | Moderate | Definite osteophytes, definite JSN |
| 4 | Severe | Large osteophytes, marked JSN, deformity |

---

## ⚕️ Disclaimer

This application is intended for **educational and decision-support purposes only**. Final diagnosis must be made by a qualified physician or radiologist.

---

## 👩‍💻 Author

**Nisma K**  
Final Year Project — B.Tech Computer Science  
Academic Year 2025-2026
