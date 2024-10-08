import nibabel as nib

# Path to your NIfTI file
file_path = "data/Second MRI dataset/CC0005_philips_15_62_M.nii.gz"

# Load the file
nii_data = nib.load(file_path)

# Get the data array (matrix)
data_matrix = nii_data.get_fdata()

print(data_matrix.shape)
# print(data_matrix)
