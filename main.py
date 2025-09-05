from fastapi import FastAPI, UploadFile, File
import os, shutil, zipfile, uuid
from processor import process_dicom_folder
from enhancer import enhance_images

app = FastAPI()

@app.post("/process/")
async def process_dicom(file: UploadFile = File(...)):
    # إنشاء معرف فريد لكل حالة
    case_id = str(uuid.uuid4())

    # تحديد المسارات
    upload_path = f"uploads/{case_id}.zip"
    extract_dir = f"uploads/{case_id}"
    output_dir = f"outputs/{case_id}"

    # إنشاء المجلدات
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # حفظ الملف المرفوع
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # فك ضغط الملف
    with zipfile.ZipFile(upload_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # تشغيل كود المعالجة الأول
    raw_images = process_dicom_folder(extract_dir, output_dir)

    # تشغيل كود التعزيز الثاني
    enhance_images(raw_images, os.path.join(output_dir, "enhanced_images"))

    # إرجاع حالة النجاح
    return {"status": "success", "case_id": case_id}
