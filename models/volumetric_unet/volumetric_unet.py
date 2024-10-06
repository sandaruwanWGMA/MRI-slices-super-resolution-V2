import torch
from monai.networks.nets import UNet

# Instantiate the 3D U-Net model from MONAI
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),  # Customize the feature map sizes
    strides=(2, 2, 2, 2),  # Pooling steps for downsampling
    num_res_units=2,  # Number of residual units per block
    norm="batch",  # Normalization type
)

# Example forward pass with a 3D volume input
input_tensor = torch.randn(
    1, 1, 64, 256, 256
)  # Example shape: (batch, channel, depth, height, width)
output = model(input_tensor)

print(output.shape)  # Output shape: (batch, out_channels, depth, height, width)

print(model)
