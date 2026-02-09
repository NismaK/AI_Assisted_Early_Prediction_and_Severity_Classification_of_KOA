import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Load DenseNet Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.densenet169(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 5)  # 5 classes 0-4
model.load_state_dict(torch.load("models/densenet169_koa.pth", map_location=device))
model.to(device)
model.eval()

# --- Transform for input images ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# --- Streamlit UI ---
st.title("AI-Assisted Knee Osteoarthritis Severity Predictor 🦵")
st.write("Upload a knee X-ray and get the predicted OA severity (0–4) with Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload a Knee X-ray (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Convert to OpenCV format
    img = np.array(image)
    img_resized = cv2.resize(img, (224,224))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # --- Prediction ---
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
    
    st.write(f"**Predicted OA Severity:** {pred_class}")
    
    # --- Grad-CAM ---
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(img_resized/255.0, grayscale_cam)
    
    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
