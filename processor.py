def process_single_dicom_file(dicom_file_path, output_dir):
    temp_dir = os.path.join(output_dir, "nifti_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # تحويل ملف DICOM إلى NIfTI
    dicom2nifti.convert_directory(os.path.dirname(dicom_file_path), temp_dir, compression=True)
    nifti_file = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz')][0]
    nifti_path = os.path.join(temp_dir, nifti_file)

    original_img = nib.load(nifti_path).get_fdata()
    image = nib.load(nifti_path)
    stripped, mask = robex(image)
    img_data = stripped.get_fdata()
    mask_data = mask.get_fdata()

    scores = [np.sum(mask_data[:, :, i]) for i in range(mask_data.shape[2])]
    top_indices = np.argsort(scores)[-3:]

    output_img_dir = os.path.join(output_dir, "raw_images")
    os.makedirs(output_img_dir, exist_ok=True)
    saved_paths = []

    for idx in sorted(top_indices):
        img_norm = (img_data[:, :, idx] - img_data.min()) / (img_data.max() - img_data.min()) * 255
        img_path = os.path.join(output_img_dir, f"slice_{idx:03d}.png")
        Image.fromarray(img_norm.astype(np.uint8)).save(img_path)
        saved_paths.append(img_path)

    return saved_paths
