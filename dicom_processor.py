import os
import dicom2nifti
import nibabel as nib
from pyrobex.robex import robex
import numpy as np
from PIL import Image
import zipfile
import shutil

def process_dicom_zip(zip_path, output_dir):
    temp_dir = os.path.join(output_dir, "temp")
    nifti_dir = os.path.join(output_dir, "nifti")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(nifti_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    dicom2nifti.convert_directory(temp_dir, nifti_dir, compression=True)
    nifti_file = next((f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')), None)
    if not nifti_file:
        raise Exception("لم يتم العثور على ملف NIfTI")

    nifti_path = os.path.join(nifti_dir, nifti_file)
    image = nib.load(nifti_path)
    original_img = image.get_fdata()

    stripped, mask = robex(image)
    img_data = stripped.get_fdata()
    mask_data = mask.get_fdata()

    scores = [np.sum(mask_data[:, :, i]) for i in range(mask_data.shape[2])]
    best_idx = int(np.argmax(scores))

    def normalize(slice):
        return ((slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(np.uint8)

    brain_slice = normalize(img_data[:, :, best_idx])
    mask_slice = (mask_data[:, :, best_idx] > 0).astype(np.uint8) * 255

    brain_path = os.path.join(output_dir, "brain.png")
    mask_path = os.path.join(output_dir, "mask.png")

    Image.fromarray(brain_slice).save(brain_path)
    Image.fromarray(mask_slice).save(mask_path)

    shutil.rmtree(temp_dir)
    shutil.rmtree(nifti_dir)

    return brain_path, mask_path
