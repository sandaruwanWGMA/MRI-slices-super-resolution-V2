import torch
import nibabel as nib
import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np

# Load a pre-trained 3D ResNet model
model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
# Modify the first convolutional layer
model.stem[0] = torch.nn.Conv3d(
    1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
)

# Load the MRI data
file_path = "data/Second MRI dataset/CC0005_philips_15_62_M.nii.gz"
nii_data = nib.load(file_path)
data_matrix = nii_data.get_fdata()

# Selecting a subset of slices, make sure it's divisible evenly by the kernel depth (3)
# Let's assume the MRI has at least 48 slices, you might want to check and adjust this
middle_index = data_matrix.shape[2] // 2
selected_slices = data_matrix[
    :, :, middle_index - 24 : middle_index + 24
]  # Select 48 slices around the middle

# Normalize the data to a range that the model expects, for example, [0, 1] or [-1, 1]
# Assuming the original data is in a medical imaging range, we can simply normalize to [0, 1]
max_val = np.amax(selected_slices)
min_val = np.amin(selected_slices)
normalized_slices = (selected_slices - min_val) / (max_val - min_val)

# Add batch and channel dimensions: [N, C, D, H, W]
input_tensor = torch.tensor(
    normalized_slices[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32
)

# Verify the shape
print("Input Tensor Shape:", input_tensor.shape)

# Feed it to the model
output = model(input_tensor)
print("Output shape from model:", output.shape)
