import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import cv2
import numpy as np
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image
import gdown, os 

# ======================================================
# 1️⃣ Auto-download DenseNet model if not exists
# ======================================================
model_path = "models/densenet169_koa.pth"
if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    url = "https://drive.google.com/file/d/12TOBCpcH0Xn0MdjhpFK6xlTu51ZVzGQ5/view?usp=sharing"  
    gdown.download(url, model_path, quiet=False)

# ======================================================
# 2️⃣ Streamlit App UI
# ======================================================
st.title("AI-Assisted Knee OA Severity Predictor 🦵")
st.write("Upload a knee X-ray and see prediction + Grad-CAM visualization.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 3️⃣ Load DenseNet model
# ======================================================
model = models.densenet169(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize([0.5],[0.5])
])

# ======================================================
# 4️⃣ Upload and predict
# ======================================================
uploaded_file = st.file_uploader("Upload Knee X-ray", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = np.array(image)
    img_resized = cv2.resize(img, (224,224))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
    st.write(f"Predicted OA Severity: {pred_class}")

    # Grad-CAM
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(img_resized/255.0, grayscale_cam)
    st.image(cam_image, caption="Grad-CAM", use_column_width=True)

    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
