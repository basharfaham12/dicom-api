from fastapi import FastAPI, UploadFile, File
import os, shutil, uuid
from processor import process_dicom_zip
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/process/")
async def process_zip(file: UploadFile = File(...)):
    case_id = str(uuid.uuid4())
    upload_dir = f"uploads/{case_id}"
    os.makedirs(upload_dir, exist_ok=True)

    zip_path = os.path.join(upload_dir, file.filename)
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image_path = process_dicom_zip(zip_path, case_id)
        return {"status": "success", "case_id": case_id, "image": os.path.basename(image_path)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
