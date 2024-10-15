import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from utils.nifti_utils import load_nifti_file, save_nifti_file
from model.gan_model import enhance_resolution
import os

# Streamlit app setup
st.title("MRI Resolution Enhancer")
st.write("Upload a low-resolution MRI NIfTI file, and click **Generate** to get a high-resolution output.")

# Upload NIfTI file
uploaded_file = st.file_uploader("Choose a low-resolution NIfTI file", type=["nii", "nii.gz"])

# Directory to save uploaded files
upload_dir = "uploads/"
os.makedirs(upload_dir, exist_ok=True)

# Check if a file has been uploaded
if uploaded_file is not None:
    # Save the uploaded file to disk
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write(f"File uploaded: {uploaded_file.name}")

    # Load the NIfTI file
    nifti_data = load_nifti_file(file_path)
    st.write("Low-resolution MRI loaded successfully.")
    
    # Display a few slices from the NIfTI file
    st.subheader("Low-Resolution MRI Preview")
    
    # Select a middle slice for visualization
    slices_to_display = [nifti_data.shape[2] // 3, nifti_data.shape[2] // 2, nifti_data.shape[2] // 3 * 2]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, slice_idx in enumerate(slices_to_display):
        axs[i].imshow(nifti_data[:, :, slice_idx], cmap="gray")
        axs[i].set_title(f"Slice {slice_idx}")
        axs[i].axis('off')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Add a button to generate the high-resolution output
    if st.button("Generate"):
        with st.spinner('Enhancing resolution...'):
            # Call the GAN model to enhance the resolution
            high_res_nifti = enhance_resolution(nifti_data)
        
        # Save the high-resolution NIfTI file
        output_file_path = os.path.join(upload_dir, "high_res_" + uploaded_file.name)
        save_nifti_file(high_res_nifti, output_file_path)

        st.success("High-resolution MRI generated successfully!")
        st.write("Download the result:")
        st.download_button("Download High-Resolution MRI", data=open(output_file_path, "rb").read(), file_name="high_res_" + uploaded_file.name)
