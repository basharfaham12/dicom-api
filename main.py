from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
from PIL import Image
import os
from image_processing import center_crop_brain, apply_clahe_and_soft_sharpen
import tensorflow as tf

app = FastAPI()

# تحميل النموذج السريري
clinical_model = tf.keras.models.load_model("assets/model/alz_model.h5")

# ترتيب الفيتشرات حسب ملف selected_features.txt
clinical_features = [
    "MMSCORE", "MMREAD", "MMWRITE", "MMDRAW", "MMHAND", "MMREPEAT",
    "MMSTATE", "MMFOLD", "MMWATCH", "MMPENCIL", "MMAREA", "MMFLOOR",
    "MMHOSPIT", "WORLDSCORE", "WORD1DL", "WORD2DL", "WORD3DL",
    "MMDAY", "MMSEASON", "AGE"
]

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


@app.post("/predict-clinical/")
async def predict_clinical(request: Request):
    data = await request.json()
    inputs = [float(data.get(key, 0.0)) for key in clinical_features]
    prediction = clinical_model.predict(np.array([inputs]))[0]
    labels = ["CN", "MCI", "AD"]
    diagnosis = labels[np.argmax(prediction)]
    return JSONResponse(content={"diagnosis": diagnosis})
