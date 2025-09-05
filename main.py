from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import os
from processor import center_crop_brain, apply_clahe_and_soft_sharpen, augment_image

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    centered = center_crop_brain(img)
    enhanced = apply_clahe_and_soft_sharpen(centered)
    resized = cv2.resize(enhanced, (224, 224))

    os.makedirs("processed", exist_ok=True)
    output_path = f"processed/{file.filename}"
    cv2.imwrite(output_path, resized)

    return JSONResponse(content={"message": "✅ تم معالجة الصورة", "path": output_path})

@app.post("/augment/")
async def augment_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    augmented = augment_image(pil_img)

    os.makedirs("augmented", exist_ok=True)
    output_path = f"augmented/aug_{file.filename}"
    augmented.save(output_path)

    return JSONResponse(content={"message": "✅ تم تعزيز الصورة", "path": output_path})
