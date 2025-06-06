
# 🌸 Flower Analyzer

**Flower Analyzer** is a FastAPI-based web application that:

- Predicts the **type** and **color** of flowers from an uploaded image.
- Estimates concentrations of essential oils: **Linalool**, **Geraniol**, and **Citronellol**.
- Includes a trained multi-output PyTorch model and a lightweight frontend.

---

## 🔧 Project Structure

```
Flower_Analyser/
├── Common/                  # Static frontend files (HTML)
├── datasets/
│   └── flower_dataset.py    # Dataset & transformations
├── models/
│   └── model.py             # PyTorch multi-output model
├── logs/                    # Runtime logs
├── best_model.pth           # Trained PyTorch model weights
├── main.py                  # FastAPI app
├── train.py                 # Model training script
└── README.md
```

---

## 🚀 Features

- Image Upload via UI or API
- Deep Learning Model with multi-output classification + regression:
  - Flower Type: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`
  - Flower Color: `white`, `yellow`, `red`, `pink`
  - Oil Concentrations: predicted in float values
- HTML UI for demo interface
- Log Tracking with timestamped logs for every prediction

---

## 🖥️ Installation

```bash
# Clone the repo
git clone https://github.com/laxparihar/Flower_Analyser.git
cd Flower_Analyser

# Install dependencies
pip install -r requirements.txt
```

> Note: `requirements.txt` should include:
> ```
> fastapi
> uvicorn
> torch
> torchvision
> Pillow
> numpy
> ```

---

## 🧠 Model Training

```bash
python train.py --data_dir data --epochs 10 --batch_size 32
```

- Make sure `data/` contains:
  - `train.csv` and `val.csv`
  - `images/` folder with the image dataset

---

## 🌐 Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

- Open in browser: [http://localhost:8080](http://localhost:8080)
- API Endpoint: `POST /predict`

---

## 📤 API Usage

### Request

```
POST /predict
Content-Type: multipart/form-data
Body: image file
```

### Response

```json
{
  "predicted_flower_type": "rose",
  "predicted_flower_color": "red",
  "estimated_oil_concentrations": {
    "Linalool": 0.32,
    "Geraniol": 0.45,
    "Citronellol": 0.27
  }
}
```

---

## ⚠️ Git Submodule Warning Fix

If you see:

```bash
warning: adding embedded git repository
```

You likely added a Git repo inside another Git repo. Fix it by:

```bash
git rm --cached Flower_Analyser
# or add it properly as a submodule:
git submodule add <url> Flower_Analyser
```

---

## 🖼️ Screenshots

### 🌼 Index Page
![Index Page](Common/Index.png)

### 🌻 Demo View
![Demo View](Common/Demo_View.png)

### 🖼️ Sample Prediction
![Sample Prediction](Common/sample.png)

