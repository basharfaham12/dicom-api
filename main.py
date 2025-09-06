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

# القيم المستخرجة من التدريب
scaler_mean = np.array([
    25.40901177446436, 0.9828513508596645, 0.9513978207223314, 0.7634146292828864,
    0.8962809832742954, 0.7550874422179567, 0.9839143914013538, 0.9792804031697696,
    0.989613783443109, 0.9935395445253702, 0.8339536896541392, 0.7612431522300387,
    0.7849456688579494, 4.356929402614527, 0.6504712853503307, 0.5250376173896225,
    0.5643060422650875, 0.7437848556504271, 0.8459203132054011, 72.02141982430106
])

scaler_std = np.array([
    4.879337034136477, 0.12630326688342508, 0.207899349710398, 0.41452493610365254,
    0.29590400646325177, 0.4210381650406388, 0.12204164152232537, 0.1393163905325267,
    0.10036738410651498, 0.07766957320428024, 0.36039393123505203, 0.41679135164540493,
    0.40115988975583755, 1.2727142669861211, 0.46910359395059603, 0.49472914086772646,
    0.49058621123444807, 0.4293772721369649, 0.35163865046095627, 7.260397380482274
])


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
