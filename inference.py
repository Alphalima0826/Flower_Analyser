import torch
from PIL import Image
from datasets.flower_dataset import get_transforms
from models.model import FlowerMultiOutputModel
import json
import argparse

def load_model(path, device):
    model = FlowerMultiOutputModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        flower_logits, color_logits, oil_preds = model(input_tensor)
        flower_idx = flower_logits.argmax(dim=1).item()
        color_idx = color_logits.argmax(dim=1).item()
        oil_preds = oil_preds.squeeze().cpu().tolist()

    flower_types = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    color_types = ["white", "yellow", "red", "pink"]

    result = {
        "predicted_flower_type": flower_types[flower_idx],
        "predicted_flower_color": color_types[color_idx],
        "estimated_oil_concentrations": {
            "Linalool": oil_preds[0],
            "Geraniol": oil_preds[1],
            "Citronellol": oil_preds[2]
        }
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to flower image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    prediction = predict(args.image, model, device)
    print(json.dumps(prediction, indent=4))

if __name__ == "__main__":
    main()