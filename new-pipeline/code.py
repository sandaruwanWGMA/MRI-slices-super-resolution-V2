import nibabel as nib
import numpy as np
import os
import glob
import SimpleITK as sitk
from scipy.ndimage import zoom

def update_header(nib_file, file_name, save_img=False, output_dir=None):

    header = nib_file.header

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

    new_header['srow_x'] = new_affine[0, :]
    new_header['srow_y'] = new_affine[1, :]
    new_header['srow_z'] = new_affine[2, :]

    updated_image = nib.Nifti1Image(nib_file.get_fdata(), new_affine, header=new_header)

    if save_img:
        nib.save(updated_image, os.path.join(output_dir, 'header_updated'+ file_name))
    
    return updated_image, file_name


def down_sample(image, file_name, factor=5, save_img=False, output_dir=None):
    
    downsampled_image = image.slicer[::factor, :, :]
    
    if save_img:
        nib.save(downsampled_image, os.path.join(output_dir, 'downsampled_' + file_name))

    return downsampled_image
    

def resample_volume(volumne_path="", interpolator=sitk.sitkLinear, new_spacing=None, output_path=""):
    if new_spacing is None:
        new_spacing = [1, 1, 1]

    voulume = sitk.ReadImage(volumne_path, sitk.sitkFloat32)
    original_spacing = voulume.GetSpacing()
    original_size = voulume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    resample_volume = sitk.Resample(voulume, new_size, sitk.Transform(), interpolator, voulume.GetOrigin(), new_spacing, voulume.GetDirection(), 0.0, voulume.GetPixelID())

    sitk.WriteImage(resample_volume, output_path)


def fix_dim(img, file_name, new_shape, save_img=False, output_dir=None):
    # Load NIfTI image
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

    if save_img:
        nib.save(resampled_img, os.path.join(output_dir, 'dimfixed_' + file_name))
    
    return resampled_img


def main():
    original_dataset = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline/DataSet"
    header_updated = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline/Header-Updated"
    resampled_highRes = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline/Resampled/High-Res"
    resampled_lowRes = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline/Resampled/Low-Res"
    downsampled = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline/Down-sampled"
    dim_fixed = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline/Dim-Fixed"

    original_data_paths = glob.glob(os.path.join(original_dataset, '*'))
    header_updated_paths = glob.glob(os.path.join(header_updated, '*'))
    dim_fixed_paths = glob.glob(os.path.join(dim_fixed, '*'))

    # For a one data point
    sample_data = original_data_paths[13]
    image = nib.load(sample_data)
    file_name = os.path.basename(sample_data)

    for i, image in enumerate(original_data_paths):
        img = nib.load(image)
        file_name = os.path.basename(image)

        dim_fixed_img = fix_dim(img, file_name, (150, 256, 256), save_img=True, output_dir=dim_fixed)

        header_updated_img, _ = update_header(dim_fixed_img, file_name, save_img=True, output_dir=header_updated)

        _ = down_sample(header_updated_img, file_name, factor=5, save_img=True, output_dir=downsampled)

        downsampled_paths = glob.glob(os.path.join(downsampled, '*'))
        header_updated_paths = glob.glob(os.path.join(header_updated, '*'))

        while len(downsampled_paths) < (i+1) or len(header_updated_paths) < (i+1):
            print(len(downsampled_paths), len(header_updated_paths))
            continue

        resample_volume(downsampled_paths[-1], output_path=os.path.join(resampled_lowRes, 'resampled_' + os.path.basename(downsampled_paths[-1])))
        resample_volume(header_updated_paths[-1], output_path=os.path.join(resampled_highRes, 'resampled_' + os.path.basename(header_updated_paths[-1])))


    # dim_fixed_img = fix_dim(image, file_name, (150, 256, 256), save_img=True, output_dir=dim_fixed)

    # header_updated_img, _ = update_header(dim_fixed_img, file_name, save_img=True, output_dir=header_updated)

    # downsampled_image = down_sample(header_updated_img, file_name, factor=5, save_img=True, output_dir=downsampled)

    # # print(downsampled_image.header)

    # downsampled_paths = glob.glob(os.path.join(downsampled, '*'))
    # header_updated_paths = glob.glob(os.path.join(header_updated, '*'))

    # while len(downsampled_paths) == 0 or len(header_updated_paths) == 0:
    #     print(len(downsampled_paths), len(header_updated_paths))
    #     continue

    # # new_image = nib.load(downsampled_paths[-1])
    # # print("Downsampled Image Header: ", new_image.header)

    # resample_volume(downsampled_paths[-1], output_path=os.path.join(resampled_lowRes, 'resampled_' + os.path.basename(downsampled_paths[-1])))
    # resample_volume(header_updated_paths[-1], output_path=os.path.join(resampled_highRes, 'resampled_' + os.path.basename(header_updated_paths[-1])))

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    # low_res_img_path = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline\Resampled\Low-Res\/resampled_downsampled_header_updateddimfixed_CC0056_philips_15_39_F.nii.gz"
    # high_res_img_path = "G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline\Resampled\High-Res\/resampled_downsampled_header_updateddimfixed_CC0056_philips_15_39_F.nii.gz"

    # resampled_low_res = nib.load(low_res_img_path)
    # resampled_high_res = nib.load(high_res_img_path)

    # print("Low Res Image Header: ", resampled_low_res.header)
    # print("High Res Image Header: ", resampled_high_res.header) 

    # resampled_low_res = nib.load("G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline\Resampled\Low-Res\/resampled_downsampled_CC0056_philips_15_39_F.nii.gz")
    # resampled_high_res = nib.load("G:\Semester 5\Data Science and Engineering Project\Codes\MRI-slices-super-resolution-V2\/new-pipeline\Resampled\High-Res\/resampled_header_updatedCC0056_philips_15_39_F.nii.gz")

    # print("Low Res Image Header: ", resampled_low_res.header)
    # print("High Res Image Header: ", resampled_high_res.header)

    main()
