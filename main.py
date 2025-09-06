from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
import cv2
from processor import center_crop_brain, apply_clahe_and_soft_sharpen

app = FastAPI()

# السماح بالوصول من تطبيق Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج بصيغة joblib
model = joblib.load("assets/model/alz_model.joblib")

# ترتيب الميزات
clinical_features = [
    "MMSCORE", "MMREAD", "MMWRITE", "MMDRAW", "MMHAND", "MMREPEAT",
    "MMSTATE", "MMFOLD", "MMWATCH", "MMPENCIL", "MMAREA", "MMFLOOR",
    "MMHOSPIT", "WORLDSCORE", "WORD1DL", "WORD2DL", "WORD3DL",
    "MMDAY", "MMSEASON", "AGE"
]

# قيم التطبيع
scaler_mean = np.array([...])  # ضع القيم هنا
scaler_std = np.array([...])   # ضع القيم هنا

# نموذج البيانات السريرية
class ClinicalData(BaseModel):
    MMSCORE: float
    MMREAD: float
    MMWRITE: float
    MMDRAW: float
    MMHAND: float
    MMREPEAT: float
    MMSTATE: float
    MMFOLD: float
    MMWATCH: float
    MMPENCIL: float
    MMAREA: float
    MMFLOOR: float
    MMHOSPIT: float
    WORLDSCORE: float
    WORD1DL: float
    WORD2DL: float
    WORD3DL: float
    MMDAY: float
    MMSEASON: float
    AGE: float

@app.post("/predict-clinical/")
async def predict_clinical(data: ClinicalData):
    inputs = np.array([getattr(data, key) for key in clinical_features])
    x_scaled = (inputs - scaler_mean) / scaler_std
    x_scaled = x_scaled.reshape(1, -1)

    preds = model.predict_proba(x_scaled)[0]
    labels = ["CN", "MCI", "AD"]
    predicted_class = labels[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {
        "diagnosis": predicted_class,
        "confidence": round(confidence * 100, 2),
        "probabilities": {labels[i]: round(float(preds[i]) * 100, 2) for i in range(len(labels))}
    }

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    centered = center_crop_brain(img)
    enhanced = apply_clahe_and_soft_sharpen(centered)
    resized = cv2.resize(enhanced, (224, 224))

    os.makedirs("processed", exist_ok=True)
    output_path = f"processed/processed_{file.filename}"
    cv2.imwrite(output_path, resized)

    return FileResponse(output_path, media_type="image/jpeg", filename=f"processed_{file.filename}")
