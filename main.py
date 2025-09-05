from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import numpy as np
from PIL import Image
import os
from image_processing import center_crop_brain, apply_clahe_and_soft_sharpen

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # قراءة الصورة من الملف المرفوع
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # المعالجة: توسيط + تحسين + ضبط الحجم
    centered = center_crop_brain(img)
    enhanced = apply_clahe_and_soft_sharpen(centered)
    resized = cv2.resize(enhanced, (224, 224))

    # حفظ الصورة المعالجة
    os.makedirs("processed", exist_ok=True)
    output_path = f"processed/processed_{file.filename}"
    cv2.imwrite(output_path, resized)

    # إعادة الصورة للطبيب مباشرة
    return FileResponse(output_path, media_type="image/jpeg", filename=f"processed_{file.filename}")
