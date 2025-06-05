from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import torch
import numpy as np
import os
import logging
import uvicorn
from io import BytesIO
from models.model import FlowerMultiOutputModel
from datasets.flower_dataset import get_transforms
from datetime import datetime
from fastapi.staticfiles import StaticFiles

today = datetime.now().strftime("%Y-%m-%d")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=f"{log_dir}/{today}.log", filemode='a', format="%(asctime)s, %(levelname)s: %(message)s")

app = FastAPI()

# Load model once at startup
device = torch.device("cpu")  # Assuming you're using CPU; change to 'cuda' if you have a GPU
model = FlowerMultiOutputModel()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

flower_types = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
color_types = ["white", "yellow", "red", "pink"]

# Mount the static files for the frontend
app.mount("/Common", StaticFiles(directory="Common"), name="Common")

@app.get("/")
def read_root():
    file_path = os.path.join(os.getcwd(), "Common", "index.html")
    return FileResponse(file_path)

@app.get("/demo")
def demo():
    file_path = os.path.join(os.getcwd(), "Common", "demo.html")
    return FileResponse(file_path)

# Load image transformation for the prediction
transform = get_transforms(train=False)

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    # Check if file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Read image bytes
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        flower_logits, color_logits, oil_preds = model(input_tensor)
        flower_idx = flower_logits.argmax(dim=1).item()
        color_idx = color_logits.argmax(dim=1).item()
        oil_preds = oil_preds.squeeze().cpu().tolist()
    
    result = {
        "predicted_flower_type": flower_types[flower_idx],
        "predicted_flower_color": color_types[color_idx],
        "estimated_oil_concentrations": {
            "Linalool": round(oil_preds[0], 2),
            "Geraniol": round(oil_preds[1], 2),
            "Citronellol": round(oil_preds[2], 2)
        }
    }
    
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
