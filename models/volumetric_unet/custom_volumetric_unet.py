# import torch
# from monai.networks.nets import UNet

# # Instantiate the 3D U-Net model from MONAI
# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm="batch",
# )

# # Example forward pass with a 3D volume input
# input_tensor = torch.randn(1, 1, 64, 256, 256)
# output = model(input_tensor)

# print("Output shape: ", output.shape)

# print(model)


import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet


class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()
        # Existing preprocessing and U-Net model code
        self.preprocess = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(5, 1, 1), stride=(5, 1, 1)),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=(2, 1, 1), padding=1),
        )
        self.unet = UNet(
            spatial_dims=3,
            in_channels=16,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
            norm="batch",
        )
        # New postprocessing layers to reshape output
        self.postprocess = nn.Sequential(
            nn.ConvTranspose3d(
                1, 1, kernel_size=(3, 3, 3), stride=(10, 2, 2), padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.Upsample(size=(150, 256, 256), mode="trilinear", align_corners=True),
        )

    def forward(self, x):
        x = self.preprocess(x)  # Preprocess to fit into the U-Net input
        x = self.unet(x)  # Process through U-Net
        x = self.postprocess(x)  # Postprocess to restore original dimensions
        return x


# Instantiate the custom model
# model = CustomUNet()

# Example input tensor
# input_tensor = torch.randn(1, 1, 150, 256, 256)

# Forward pass
# output = model(input_tensor)

# print("Output shape:", output.shape)

# print(model)
