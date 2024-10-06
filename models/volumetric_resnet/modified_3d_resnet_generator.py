import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F


class Modified3DResNet(nn.Module):
    def __init__(self):
        super(Modified3DResNet, self).__init__()
        # Load a pre-trained 3D ResNet model
        self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Modify the first convolutional layer to accept 1 channel (instead of 3)
        self.model.stem[0] = nn.Conv3d(
            in_channels=1,  # Change input channels to 1 for single-channel MRI input
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )

        # Do not remove the avgpool to avoid flattening the tensor prematurely
        # We'll adjust the final layers after the avgpool
        self.model.fc = nn.Identity()  # Bypass the final fully connected layer

        # Add a 3D convolutional layer to reduce the output channels to 1
        self.conv_to_single_channel = nn.Conv3d(
            in_channels=512,  # The output of the ResNet is 512 channels
            out_channels=1,  # Reduce channels to 1
            kernel_size=1,  # 1x1x1 convolution to change channels only
        )

        # Add an upsampling layer to increase the depth to 150
        self.upsample_layer = nn.Upsample(
            size=(150, 256, 256), mode="trilinear", align_corners=False
        )

    def forward(self, x):
        # Pass the input through the ResNet model up to the avgpool layer
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)  # Preserve spatial structure

        # Apply the 1x1x1 convolution to reduce channels to 1
        x = self.conv_to_single_channel(x)

        # Upsample the depth from whatever it is to 150
        x = self.upsample_layer(x)

        return x


# Example usage
def process_mri_data(file_path):
    import nibabel as nib
    import numpy as np

    # Load the MRI data
    nii_data = nib.load(file_path)
    data_matrix = nii_data.get_fdata()

    # Reorder dimensions if necessary: MRI might be in (H, W, D), we want (D, H, W)
    data_matrix = np.transpose(data_matrix, (2, 0, 1))  # Reorder to [D, H, W]

    # Select the first 30 slices if available (adjust as needed)
    num_slices = 30
    selected_slices = data_matrix[:num_slices, :, :]

    # Resize the height and width to 256x256 using torch's interpolate (if not already the size)
    selected_slices = torch.tensor(
        selected_slices[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32
    )
    selected_slices = F.interpolate(
        selected_slices,
        size=(num_slices, 256, 256),
        mode="trilinear",
        align_corners=False,
    )

    return selected_slices


# Instantiate the modified ResNet model
model = Modified3DResNet()

# Load the MRI data and convert to a tensor
file_path = "data/Second MRI dataset/CC0005_philips_15_62_M.nii.gz"
input_tensor = process_mri_data(file_path)

# Pass the input through the modified model
output = model(input_tensor)

# Verify the shapes
print("Input shape:", input_tensor.shape)  # Should be [1, 1, 30, 256, 256]
print("Output shape:", output.shape)  # Should be [1, 1, 150, 256, 256]

print(model)
