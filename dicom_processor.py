@app.post("/process-case/")
async def process_case(zip_file: UploadFile = File(...)):
    try:
        import uuid
        import os
        import zipfile
        import dicom2nifti
        import nibabel as nib
        from pyrobex.robex import robex
        import numpy as np
        from PIL import Image

        # إنشاء مجلد مؤقت للحالة
        case_id = str(uuid.uuid4())
        temp_dir = f"temp_cases/{case_id}"
        os.makedirs(temp_dir, exist_ok=True)

        # حفظ الملف المضغوط
        zip_path = os.path.join(temp_dir, "case.zip")
        with open(zip_path, "wb") as f:
            f.write(await zip_file.read())

        # فك ضغط ملفات DICOM
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # تحويل DICOM إلى NIfTI
        nifti_out = os.path.join(temp_dir, "nifti")
        os.makedirs(nifti_out, exist_ok=True)
        dicom2nifti.convert_directory(temp_dir, nifti_out, compression=True)
        nifti_file = [f for f in os.listdir(nifti_out) if f.endswith('.nii.gz')][0]
        nifti_path = os.path.join(nifti_out, nifti_file)

        # تحميل الصورة الأصلية
        image = nib.load(nifti_path)
        stripped, mask = robex(image)
        img_data = stripped.get_fdata()
        mask_data = mask.get_fdata()

        # اختيار أفضل شريحة واحدة
        scores = [np.sum(mask_data[:, :, i]) for i in range(mask_data.shape[2])]
        best_idx = int(np.argmax(scores))

        # تجهيز الصورة والقناع
        def normalize(slice):
            return ((slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(np.uint8)

        brain_slice = normalize(img_data[:, :, best_idx])
        mask_slice = (mask_data[:, :, best_idx] > 0).astype(np.uint8) * 255

        # حفظ الصورة والقناع
        image_path = os.path.join("saved_outputs", f"{case_id}_brain.png")
        mask_path = os.path.join("saved_masks", f"{case_id}_mask.png")

        Image.fromarray(brain_slice).save(image_path)
        Image.fromarray(mask_slice).save(mask_path)

        return JSONResponse(content={
            "case_id": case_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "status": "success"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
