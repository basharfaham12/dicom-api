# processor.py
import os
import dicom2nifti
import nibabel as nib
from pyrobex.robex import robex
from PIL import Image
import numpy as np

def process_dicom_folder(base_dir, output_dir):
    output_img_dir = os.path.join(output_dir, "raw_images")
    os.makedirs(output_img_dir, exist_ok=True)
    saved_paths = []

    def save_slice(img_slice, mask_slice, orig_slice, case_id, slice_idx):
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
        img_path = os.path.join(output_img_dir, f"{case_id}_slice_{slice_idx:03d}.png")
        Image.fromarray(img_norm.astype(np.uint8)).save(img_path)
        saved_paths.append(img_path)

    case_folders = sorted(os.listdir(base_dir))
    for case_id in case_folders:
        dicom_path = os.path.join(base_dir, case_id)
        nifti_out = os.path.join(output_dir, "nifti_temp", case_id)
        os.makedirs(nifti_out, exist_ok=True)

        try:
            dicom2nifti.convert_directory(dicom_path, nifti_out, compression=True)
            nifti_file = [f for f in os.listdir(nifti_out) if f.endswith('.nii.gz')][0]
            nifti_path = os.path.join(nifti_out, nifti_file)

            original_img = nib.load(nifti_path).get_fdata()
            image = nib.load(nifti_path)
            stripped, mask = robex(image)
            img_data = stripped.get_fdata()
            mask_data = mask.get_fdata()

            scores = [np.sum(mask_data[:, :, i]) for i in range(mask_data.shape[2])]
            top_indices = np.argsort(scores)[-3:]

            for idx in sorted(top_indices):
                save_slice(img_data[:, :, idx], mask_data[:, :, idx], original_img[:, :, idx], case_id, idx)

        except Exception as e:
            print(f"❌ فشل في معالجة الحالة: {case_id} - {e}")

    return saved_paths
