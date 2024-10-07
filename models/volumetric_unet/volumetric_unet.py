import torch
from monai.networks.nets import UNet

# Instantiate the 3D U-Net model from MONAI
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
)

# Example forward pass with a 3D volume input
input_tensor = torch.randn(1, 1, 64, 256, 256)
output = model(input_tensor)

print(output.shape)

print(model)
