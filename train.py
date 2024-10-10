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
from data.datasets import get_dataloader


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

    # Data loader
    dataloader = get_dataloader()
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            real_images = data.to(device)

            # ===================
            # Update discriminator
            # ===================
            discriminator.zero_grad()
            # Train with real MRI images
            real_pred = discriminator(real_images)
            loss_D_real = criterion(real_pred, True)
            # Train with fake MRI images
            fake_images = generator(
                torch.randn(real_images.size(0), 1, 30, 256, 256, device=device)
            )
            fake_pred = discriminator(fake_images.detach())
            loss_D_fake = criterion(fake_pred, False)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()

            # =================
            # Update generator
            # =================
            generator.zero_grad()
            fake_pred = discriminator(fake_images)
            loss_G = criterion(fake_pred, True)
            loss_G.backward()
            opt_G.step()

            # Logging
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
                )

            # Update learning rate
            scheduler_G.step()
            scheduler_D.step()

    # Save models for later use
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")


if __name__ == "__main__":
    main()
