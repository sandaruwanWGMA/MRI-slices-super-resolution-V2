import nibabel as nib

def load_nifti_file(file_path):
    """Load a NIfTI file from the given file path."""
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    return nifti_data

def save_nifti_file(nifti_data, file_path):
    """Save a NIfTI file to the given file path."""
    new_img = nib.Nifti1Image(nifti_data, affine=np.eye(4))  # Identity affine matrix for simplicity
    nib.save(new_img, file_path)
