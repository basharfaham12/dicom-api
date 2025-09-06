from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
import os
from image_processing import center_crop_brain, apply_clahe_and_soft_sharpen
app = FastAPI()

# تحميل نموذج TFLite
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="assets/model/alz_mlp_model.tflite")
interpreter.allocate_tensors()


# ترتيب الفيتشرات حسب ملف selected_features.txt
clinical_features = [
    "MMSCORE", "MMREAD", "MMWRITE", "MMDRAW", "MMHAND", "MMREPEAT",
    "MMSTATE", "MMFOLD", "MMWATCH", "MMPENCIL", "MMAREA", "MMFLOOR",
    "MMHOSPIT", "WORLDSCORE", "WORD1DL", "WORD2DL", "WORD3DL",
    "MMDAY", "MMSEASON", "AGE"
]

@app.post("/predict-clinical/")
async def predict_clinical(request: Request):
    data = await request.json()

    # تجهيز المدخلات بالترتيب الصحيح
    inputs = [float(data.get(key, 0.0)) for key in clinical_features]
    input_data = np.array([inputs], dtype=np.float32)

    # تشغيل النموذج
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)[0]

    # استخراج التصنيف
    labels = ["CN", "MCI", "AD"]
    diagnosis = labels[np.argmax(output_data)]

    return JSONResponse(content={"diagnosis": diagnosis})


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
