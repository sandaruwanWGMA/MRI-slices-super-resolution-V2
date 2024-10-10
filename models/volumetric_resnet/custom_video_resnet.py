import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights


class Modified3DResNet(nn.Module):
    def __init__(self, pretrained=True, freeze_pretrained=True):
        super(Modified3DResNet, self).__init__()
        # Load the pre-trained 3D ResNet model
        if pretrained:
            model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            model = r3d_18(pretrained=False)

        # Modify the first convolutional layer in the stem to accept 1 input channel
        model.stem[0] = nn.Conv3d(
            1,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )

        # Use initial layers from the pretrained model
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        if freeze_pretrained:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False

        # Additional layers to reshape output to [1, 1, 150, 256, 256]
        self.upsample1 = nn.ConvTranspose3d(
            256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.upsample2 = nn.ConvTranspose3d(
            128, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.upsample3 = nn.ConvTranspose3d(
            64, 1, kernel_size=(1, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(self.upsample1(x))
        x = F.relu(self.upsample2(x))
        x = self.upsample3(x)

        # Ensure the output shape matches the desired dimensions
        x = F.interpolate(
            x, size=(150, 256, 256), mode="trilinear", align_corners=False
        )

        return x


# modified_model = Modified3DResNet()

# print(modified_model)
