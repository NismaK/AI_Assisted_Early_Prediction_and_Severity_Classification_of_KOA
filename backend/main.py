from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import json
import os
import cv2
import base64
from datetime import datetime
from report import generate_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NUM_CLASSES = 5
LABELS = ['Normal', 'Doubtful', 'Mild OA', 'Moderate OA', 'Severe OA']
device = torch.device('cpu')
HISTORY_FILE = "patient_history.json"

# ── Patient history utilities ─────────────────────────────────
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

# ── Model loaders ─────────────────────────────────────────────
def load_efficientnet_b5():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(
        'efficientnet_b5_best.pt', map_location=device))
    model.eval()
    return model

def load_efficientnet_v2_s():
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(
        'efficientnet_v2_s_best.pt', map_location=device))
    model.eval()
    return model

def load_densenet_201():
    model = models.densenet201(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(
        'densenet_201_best.pt', map_location=device))
    model.eval()
    return model

print("Loading models...")
models_list = [
    load_efficientnet_b5(),
    load_efficientnet_v2_s(),
    load_densenet_201(),
]
print("All models ready!")

# ── Image transform ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Grad-CAM implementation ───────────────────────────────────
def generate_gradcam(model, image_tensor, target_class):
    """Generate Grad-CAM heatmap from the last conv layer."""
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # Hook the last feature layer
    if hasattr(model, 'features'):
        target_layer = model.features[-2]
    else:
        target_layer = list(model.children())[-2]

    handle = target_layer.register_forward_hook(save_activation)

    # Forward pass with gradients enabled
    model.eval()
    output = model(image_tensor)

    # Backward on predicted class score
    model.zero_grad()
    output[0, target_class].backward()
    handle.remove()

    # Compute weighted activation map
    grad = gradients[0].squeeze().cpu().detach().numpy()
    act = activations[0].squeeze().cpu().detach().numpy()

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > 0:
        cam = cam / cam.max()

    # Apply jet colormap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap

def overlay_heatmap_on_image(image_bytes, heatmap, alpha=0.45):
    """Blend Grad-CAM heatmap onto original X-ray and return base64."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    blended = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)

    _, buffer = cv2.imencode(
        '.jpg', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Knee OA backend is running!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form("Unknown"),
    patient_age: str = Form(""),
    patient_gender: str = Form("Male"),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # ── Ensemble inference ────────────────────────────────────
    all_probs = []
    with torch.no_grad():
        for model in models_list:
            out = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)
            all_probs.append(out.numpy())

    avg_probs = np.mean(all_probs, axis=0)[0]
    grade = int(np.argmax(avg_probs))
    confidence = int(avg_probs[grade] * 100)

    # ── Grad-CAM from EfficientNet-B5 ────────────────────────
    tensor_gc = transform(image).unsqueeze(0)
    tensor_gc.requires_grad_(True)
    heatmap = generate_gradcam(models_list[0], tensor_gc, grade)
    gradcam_b64 = overlay_heatmap_on_image(contents, heatmap)

    findings = {
        0: ['Joint space normal', 'No osteophytes present', 'Normal bone density'],
        1: ['Possible osteophyte', 'Joint space doubtful', 'Early changes possible'],
        2: ['Definite osteophytes present', 'Joint space mildly reduced', 'Mild sclerosis'],
        3: ['Multiple osteophytes', 'Joint space significantly reduced', 'Definite sclerosis'],
        4: ['Large osteophytes', 'Joint space severely reduced', 'Severe sclerosis and deformity'],
    }

    all_probabilities = {
        f"Grade {i}": round(float(avg_probs[i]) * 100, 1)
        for i in range(NUM_CLASSES)
    }

    save_to_history(
        {"name": patient_name, "age": patient_age, "gender": patient_gender},
        {"grade": grade, "label": LABELS[grade],
         "confidence": confidence, "findings": findings[grade]}
    )

    return {
        "grade": grade,
        "label": LABELS[grade],
        "confidence": confidence,
        "findings": findings[grade],
        "all_probabilities": all_probabilities,
        "gradcam_image": gradcam_b64,
    }

@app.post("/generate-report")
async def create_report(data: dict):
    filepath = generate_report(data['patient'], data['result'])
    return FileResponse(
        filepath,
        media_type='application/pdf',
        filename="knee_oa_report.pdf"
    )

@app.get("/history")
def get_history():
    return load_history()

@app.delete("/history/{record_id}")
def delete_record(record_id: int):
    history = load_history()
    history = [r for r in history if r["id"] != record_id]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    return {"message": "Record deleted successfully"}