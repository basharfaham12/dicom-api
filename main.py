from fastapi import FastAPI, UploadFile, File
import os, shutil, uuid
from processor import process_single_dicom_file
from enhancer import enhance_images

app = FastAPI()

@app.post("/process_single/")
async def process_single_dicom(file: UploadFile = File(...)):
    case_id = str(uuid.uuid4())
    upload_dir = f"uploads/{case_id}"
    os.makedirs(upload_dir, exist_ok=True)

    dicom_path = os.path.join(upload_dir, file.filename)
    with open(dicom_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_dir = f"outputs/{case_id}"
    os.makedirs(output_dir, exist_ok=True)

    raw_images = process_single_dicom_file(dicom_path, output_dir)
    enhance_images(raw_images, os.path.join(output_dir, "enhanced_images"))

    return {"status": "success", "case_id": case_id}
