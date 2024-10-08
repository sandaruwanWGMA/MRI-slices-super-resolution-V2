import nibabel as nib
import numpy as np
import os
import glob
import SimpleITK as sitk
import torch
from monai.transforms import Resize
from scipy.ndimage import zoom


def read_nifti_file(filepath, save_header=False):
    # Extract the file name from the file path
    file_name = os.path.basename(filepath)

    # Read the .nii file
    img = nib.load(filepath)

    # Extract header data
    header = img.header

    # Extract image data
    data = img.get_fdata()

    # Save the header data if specified
    if save_header:
        with open('./data/' + file_name + '.txt', 'w') as file:
            file.write(str(header))
    
    return img, header, data, file_name


# Function to resample NIfTI image and update the header and affine matrix
def resample_nifti_monai_with_affine(nifti_file, new_shape):
    # Load the NIfTI image
    img = nib.load(nifti_file)
    data = img.get_fdata()
    original_shape = data.shape
    original_affine = img.affine

    # Create a tensor from the image data and add batch and channel dimensions
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # Define the MONAI Resize transform
    resize_transform = Resize(spatial_size=new_shape, mode='area')  # mode='area' for downsampling

    # Apply the transform to resample the image
    resampled_tensor = resize_transform(data_tensor)

    # Remove extra dimensions to get the resampled data
    resampled_data = resampled_tensor.squeeze().numpy()

    # Compute the scaling factors based on the old and new shapes
    scaling_factors = np.array(original_shape) / np.array(new_shape)

    # Update the affine matrix by adjusting the voxel size (scaling)
    new_affine = original_affine.copy()
    for i in range(3):  # Only scale the spatial dimensions (x, y, z)
        new_affine[i, i] *= scaling_factors[i]

    # Create a new NIfTI image with the resampled data and the new affine matrix
    resampled_img = nib.Nifti1Image(resampled_data, new_affine, header=img.header)

    # Update the header to reflect the new voxel sizes (pixel dimensions)
    # The voxel size is stored in the header's `pixdim` field
    new_voxel_sizes = np.array(img.header.get_zooms()) * scaling_factors
    resampled_img.header.set_zooms(new_voxel_sizes)

    return resampled_img


# Function to resample NIfTI image to a fixed shape
def resample_nifti(nifti_file, new_shape):
    # Load NIfTI image
    img = nib.load(nifti_file)
    data = img.get_fdata()
    
    # Original shape and voxel dimensions
    original_shape = data.shape
    original_affine = img.affine
    
    # Calculate the rescaling factors for each dimension
    scaling_factors = [new_dim / old_dim for new_dim, old_dim in zip(new_shape, original_shape)]
    
    # Resample the image data to the new shape
    resampled_data = zoom(data, scaling_factors, order=1)  # 'order=1' for linear interpolation
    
    # Create new affine by scaling the voxel sizes
    new_affine = original_affine.copy()
    new_affine[:3, :3] *= np.array([original_shape[i] / new_shape[i] for i in range(3)])
    
    # Create new NIfTI image with resampled data and updated affine
    resampled_img = nib.Nifti1Image(resampled_data, new_affine)
    
    return resampled_img


# Function to update the header and affine matrix
def overide_header(img, header):
    voxel_spacing = header.get_zooms()

    qoffset_x = header['qoffset_x']
    qoffset_y = header['qoffset_y']
    qoffset_z = header['qoffset_z']

    new_affine = np.array([
        [voxel_spacing[0], 0, 0, qoffset_x],
        [0, voxel_spacing[1], 0, qoffset_y],
        [0, 0, voxel_spacing[2], qoffset_z],
        [0, 0, 0, 1]
    ])

    new_header = header.copy()
    
    new_header['srow_x'] = new_affine[0,:]
    new_header['srow_y'] = new_affine[1,:]
    new_header['srow_z'] = new_affine[2,:]

    new_image = nib.Nifti1Image(img.get_fdata(), new_affine, new_header)

    return new_image


def downsample_image(img):
    return img.slicer[::5, :, :]


def convert_voxel(img, file_name, new_spacing=[1, 1, 1], downsampled=False):
    interpolator = sitk.sitkLinear

    img_data = img.get_fdata()

    volume = sitk.GetImageFromArray(img_data.astype(np.float32))

    old_spacing = volume.GetSpacing()
    old_size = volume.GetSize()

    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)]
    resample_volume = sitk.Resample(volume, new_size, sitk.Transform(), interpolator, volume.GetOrigin(), new_spacing, volume.GetDirection(), 0.0, volume.GetPixelID())

    if downsampled:
        name = f"{file_name}_down_resampled.nii.gz"
    else:
        name = f"{file_name}_resampled.nii.gz"

    sitk.WriteImage(resample_volume, name)


if __name__ == '__main__':
    base_dir = r"G:\Semester 5\Data Science and Engineering Project\Codes\MRI-Enhance-3D\DataSet"
    file_paths = glob.glob(os.path.join(base_dir, '*'))

    sample_file = file_paths[2]

    img, header, data, file_name = read_nifti_file(sample_file)

    resampled_img = resample_nifti(sample_file, (150, 256, 256))

    new_img = overide_header(resampled_img, header)

    down_sampled_img = downsample_image(new_img)

    convert_voxel(new_img, file_name)
    convert_voxel(down_sampled_img, file_name, downsampled=True)

