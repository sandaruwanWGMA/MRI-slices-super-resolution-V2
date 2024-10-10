import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms


# Import your model definitions
from models.volumetric_resnet.custom_video_resnet import Modified3DResNet
from models.volumetric_unet.custom_volumetric_unet import CustomUNet

# Import utility functions
from util.losses import GANLoss, cal_gradient_penalty
from util.schedulers import get_scheduler

# Import Custom Dataset
from data.dataloader import MRIDataset


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    device = setup_device()

    # Initialize the generator and discriminator
    generator = Modified3DResNet().to(device)
    discriminator = CustomUNet().to(device)

    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Losses
    criterion = GANLoss(gan_mode="lsgan").to(device)

    # Learning rate schedulers
    scheduler_G = get_scheduler(opt_G, {"lr_policy": "step", "lr_decay_iters": 10})
    scheduler_D = get_scheduler(opt_D, {"lr_policy": "step", "lr_decay_iters": 10})

    # Define the path to the base directory containing both Low-Res and High-Res directories
    base_dir = "./MRI Dataset"

    # Create the dataset and dataloader
    mri_dataset = MRIDataset(base_dir)
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            print(f"epoch number: {epoch} and i: {i}")
            high_res_images = data[1].to(device)
            low_res_images = data[1].to(device)

            # ===================
            # Update discriminator
            # ===================
            discriminator.zero_grad()
            # Train with real MRI images
            real_pred = discriminator(high_res_images)
            loss_D_real = criterion(real_pred, True)
            # Train with fake MRI images
            fake_images = generator(low_res_images)
            # fake_pred = discriminator(fake_images.detach())


if __name__ == "__main__":
    main()
