import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## HW05 임자이")
st.write("🐰")
st.sidebar.write("## my sidebar")

# Image processing function
def process_image(upload):
    image = Image.open(upload)

    # Part 1: Basic image processing
    st.subheader("1. Basic Image Processing")

    col1, col2, col3, col4 = st.columns(4)

    col1.write("Input")
    col1.image(image, use_container_width=True)

    output = image.convert('L')
    col2.write("Output 1: grayscale")
    col2.image(output, use_container_width=True)

    output = ImageOps.equalize(output)
    col3.write("Output 2: histeq")
    col3.image(output, use_container_width=True)

    output = np.max(np.array(output)) - np.array(output)
    col4.write("Output 3: image inversion")
    col4.image(output, clamp=True, use_container_width=True)

    # Part 2: Object counting
    st.subheader("2. Object Counting")

    # Convert image to grayscale array
    I = image.convert('L')
    f = np.array(I)

    # Median filtering
    f_denoised = signal.medfilt2d(f, 3)

    col1, col2 = st.columns(2)

    col1.write("Original Grayscale Image")
    col1.image(I, use_container_width=True)

    col2.write("Median Filtered Image")
    col2.image(f_denoised, clamp=True, use_container_width=True)

    # Histogram
    st.write("Histogram of Median Filtered Image")

    fig, ax = plt.subplots()
    ax.hist(
        f_denoised.ravel(),
        bins=256,
        range=(0, 256),
        color='gray'
    )
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

    # Thresholding
    th_val = st.sidebar.slider(
        "Threshold Value",
        min_value=0,
        max_value=255,
        value=128
    )

    f_mask = f_denoised > th_val

    # Connected component labeling
    labels, nums = nd.label(f_mask)

    st.subheader("3. Thresholding & Labeling")

    col1, col2 = st.columns(2)

    fig1, ax1 = plt.subplots()
    ax1.imshow(f_mask, cmap='gray')
    ax1.axis('off')
    ax1.set_title("Binary Image")
    col1.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.imshow(labels, cmap='Spectral_r')
    ax2.axis('off')
    ax2.set_title("Connected Component Labeling")
    col2.pyplot(fig2)
    plt.close(fig2)

    # Object counting
    st.subheader("4. Object Counting Results")

    st.write(f"Total number of objects: {nums}")

    # Calculate component sizes
    cc = np.zeros((nums, 2))

    for i in range(1, nums + 1):
        cc[i-1, 0] = i
        cc[i-1, 1] = np.sum(labels == i)

    if nums > 0:
        ind_max = np.argmax(cc[:, 1])
        ind_min = np.argmin(cc[:, 1])

        max_label = int(cc[ind_max, 0])
        min_label = int(cc[ind_min, 0])

        st.write(
            f"Largest component: index {max_label}, "
            f"number of pixels: {int(cc[ind_max, 1])}"
        )

        st.write(
            f"Smallest component: index {min_label}, "
            f"number of pixels: {int(cc[ind_min, 1])}"
        )

        # Visualization
        col1, col2, col3 = st.columns(3)

        fig, ax = plt.subplots()
        ax.imshow(f, cmap='gray')
        ax.axis('off')
        ax.set_title("Input Image")
        col1.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.imshow(labels == max_label, cmap='gray')
        ax.axis('off')
        ax.set_title("Largest Component")
        col2.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.imshow(labels == min_label, cmap='gray')
        ax.axis('off')
        ax.set_title("Smallest Component")
        col3.pyplot(fig)
        plt.close(fig)

        # Component information
        st.subheader("5. Component Information")

        st.dataframe(
            {
                "Component Index": cc[:, 0].astype(int),
                "Number of Pixels": cc[:, 1].astype(int)
            },
            hide_index=True
        )

    else:
        st.warning("No objects detected. Try adjusting the threshold.")


# Image upload
my_upload = st.sidebar.file_uploader(
    "Upload",
    type=["png", "jpg", "jpeg"]
)

if my_upload is not None:
    process_image(upload=my_upload)
else:
    image_path = Path(__file__).resolve().parent / ""
    process_image(str(image_path))
