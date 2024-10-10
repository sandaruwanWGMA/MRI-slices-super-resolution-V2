import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os


class MRIGANDataset(Dataset):
    def __init__(self, data_dir, gen_slices=30, disc_slices=150, transform=None):
        """
        Args:
            data_dir (str): Directory with all the NIfTI files.
            gen_slices (int): Number of slices for the generator input.
            disc_slices (int): Number of slices for the discriminator input.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        # Filter files to include only .nii or .nii.gz
        self.files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.endswith(".nii") or f.endswith(".nii.gz")
            ]
        )
        self.gen_slices = gen_slices
        self.disc_slices = disc_slices
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        img = nib.load(file_path).get_fdata()

        # Ensure the image has enough slices for both generator and discriminator
        if img.shape[2] < max(self.gen_slices, self.disc_slices):
            raise ValueError(f"Not enough slices in file {self.files[idx]}")

        # Select the required number of slices for generator and discriminator
        gen_data = img[:, :, : self.gen_slices]
        disc_data = img[:, :, : self.disc_slices]

        # Optionally, apply any transform (e.g., resizing or normalization)
        if self.transform:
            gen_data = self.transform(gen_data)
            disc_data = self.transform(disc_data)

        # Convert to torch tensors and adjust dimensions to [Channels, Depth, Height, Width]
        gen_data = torch.tensor(gen_data).unsqueeze(0).permute(0, 2, 1, 3)
        disc_data = torch.tensor(disc_data).unsqueeze(0).permute(0, 2, 1, 3)

        return gen_data, disc_data


if __name__ == "__main__":
    # Define the data directory and create the dataset
    data_dir = "./MRI Dataset"  # Replace with your actual path
    dataset = MRIGANDataset(
        data_dir=data_dir, gen_slices=30, disc_slices=150, transform=None
    )

    # Create the DataLoader with num_workers=0 to avoid multiprocessing issues on macOS
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Test the DataLoader by iterating through a batch
    for gen_data, disc_data in data_loader:
        print(
            "Generator input shape:", gen_data.shape
        )  # Expected: [N, 1, 30, 256, 256]
        print(
            "Discriminator input shape:", disc_data.shape
        )  # Expected: [N, 1, 150, 256, 256]
        break
