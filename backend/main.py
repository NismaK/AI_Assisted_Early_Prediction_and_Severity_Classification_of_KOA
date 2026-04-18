from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from report import generate_report  
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form
app = FastAPI()
import json
import os
from datetime import datetime

HISTORY_FILE = "patient_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_to_history(patient, result):
    history = load_history()
    record = {
        "id": len(history) + 1,
        "date": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "patient": patient,
        "result": {
            "grade": result["grade"],
            "label": result["label"],
            "confidence": result["confidence"],
            "findings": result["findings"],
        }
    }
    history.append(record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    return record

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NUM_CLASSES = 5
CLASS_NAMES = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
LABELS = ['Normal', 'Doubtful', 'Mild OA', 'Moderate OA', 'Severe OA']
device = torch.device('cpu')

def load_efficientnet_b5():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load('efficientnet_b5_best.pt', map_location=device))
    model.eval()
    return model

def load_efficientnet_v2_s():
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load('efficientnet_v2_s_best.pt', map_location=device))
    model.eval()
    return model

def load_densenet_201():
    model = models.densenet201(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load('densenet_201_best.pt', map_location=device))
    model.eval()
    return model

print("Models are loading...")
models_list = [
    load_efficientnet_b5(),
    load_efficientnet_v2_s(),
    load_densenet_201(),
]
print("All models are ready!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@app.get("/")
def root():
    return {"status": "Knee OA Backend is running!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form("Unknown"),
    patient_age: str = Form(""),
    patient_gender: str = Form("Male"),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    all_probs = []
    with torch.no_grad():
        for model in models_list:
            out = torch.softmax(model(tensor), dim=1)
            all_probs.append(out.numpy())

    avg_probs = np.mean(all_probs, axis=0)[0]
    grade = int(np.argmax(avg_probs))
    confidence = int(avg_probs[grade] * 100)

    findings = {
        0: ['Joint space is normal', 'There is no osteophyte present', 'Normal bone density'],
        1: ['Possible osteophyte', 'Joint space doubtful', 'Early changes possible'],
        2: ['Definite osteophytes hain', 'Joint space is less', 'Mild sclerosis'],
        3: ['Multiple osteophytes', 'Joint space is significantly less', 'Definite sclerosis'],
        4: ['Large osteophytes', 'Joint space is severely less', 'Severe sclerosis and deformity'],
    }

    all_probabilities = {
        f"Grade {i}": round(float(avg_probs[i]) * 100, 1)
        for i in range(NUM_CLASSES)
    }

    # History mein save karo
    save_to_history(
        {"name": patient_name, "age": patient_age, "gender": patient_gender},
        {
            "grade": grade,
            "label": LABELS[grade],
            "confidence": confidence,
            "findings": findings[grade],
        }
    )

    return {
        "grade": grade,
        "label": LABELS[grade],
        "confidence": confidence,
        "findings": findings[grade],
        "all_probabilities": all_probabilities,
    }
@app.post("/generate-report")
async def create_report(data: dict):
    filepath = generate_report(data['patient'], data['result'])
    return FileResponse(filepath, media_type='application/pdf',
                       filename=f"knee_oa_report.pdf")
@app.get("/history")
def get_history():
    return load_history()

@app.delete("/history/{record_id}")
def delete_record(record_id: int):
    history = load_history()
    history = [r for r in history if r["id"] != record_id]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    return {"message": "Record deleted"}