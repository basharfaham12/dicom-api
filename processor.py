import os, zipfile
import dicom2nifti
import nibabel as nib
from pyrobex.robex import robex
from PIL import Image
import numpy as np

def process_dicom_zip(zip_path, case_id):
    extract_dir = f"temp/{case_id}/dicom"
    nifti_dir = f"temp/{case_id}/nifti"
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(nifti_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    dicom2nifti.convert_directory(extract_dir, nifti_dir, compression=True)
    nifti_file = [f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')][0]
    nifti_path = os.path.join(nifti_dir, nifti_file)

    image = nib.load(nifti_path)
    stripped, mask = robex(image)
    img_data = stripped.get_fdata()
    mask_data = mask.get_fdata()

    scores = [np.sum(mask_data[:, :, i]) for i in range(mask_data.shape[2])]
    best_idx = int(np.argmax(scores))
    best_slice = img_data[:, :, best_idx]
    norm_slice = (best_slice - best_slice.min()) / (best_slice.max() - best_slice.min()) * 255

    output_path = f"outputs/{case_id}_brain.png"
    Image.fromarray(norm_slice.astype(np.uint8)).save(output_path)
    return output_path
