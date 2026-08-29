
## python "C:\Users\super\Documents\projects\Yayscires\zcode\rawImageAlignment.py" "C:\\Users\\super\\Documents\\projects\\Yayscires\\raw_imgs\\" 10

import os
import sys
import cv2
import numpy as np
from PIL import Image
from glob import glob
import piexif
from multiprocessing import Pool # boooommm
from itertools import repeat
import time
from shutil import copy as copy_file


""" Settings """
# Define termination criteria for the image registration algorithm; empirically determined:
number_of_iterations = 100
termination_eps = 1e-6

"""Main"""
def load_image(filename):
    """Load a single image with the metadata as byte dump"""
    image = np.asarray(Image.open(filename))
    # metadata
    exif_dict = piexif.load(filename)
    metadata = piexif.dump(exif_dict) # dump as bytes
    return image, metadata

def save_image(filename, image, metadata):
    """ Save the image with the original metadata """
    image_arr = Image.fromarray(image)
    image_arr.save(filename, "tiff", exif=metadata)

def align_images_rgb(image1, image2, metadata2, img_save_name):
    """Image registration for RGB images via MOTION_HOMOGRAPHY."""
    # Check if the input images are multi-channel (RGB)
    if len(image1.shape) == 3 and len(image2.shape) == 3:
        # Split RGB channels
        channels1 = cv2.split(image1)
        channels2 = cv2.split(image2)

        # Aligned channels will be stored here
        aligned_channels = []

        # Parameters
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        sz = image1.shape[:2]  # Height and width only (for RGB images)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Process each channel independently
        for ch1, ch2 in zip(channels1, channels2):
            ch1_float = ch1.astype(np.float32)
            ch2_float = ch2.astype(np.float32)

            # Speed up the computation of the homography matrix by using scaled-down images
            n_level = 4  # number of scaling levels (pyramid size)
            for level in range(n_level):
                scale = 1 / 2**(n_level - 1 - level)
                rszImg1 = cv2.resize(ch1_float, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
                rszImg2 = cv2.resize(ch2_float, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)

                # Run the ECC algorithm. The results are stored in warp_matrix.
                _, warp_matrix = cv2.findTransformECC(rszImg1, rszImg2, warp_matrix, warp_mode, criteria, inputMask=None)

                if level != n_level - 1:  # Scale up for the next pyramid level
                    warp_matrix = warp_matrix * np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [1.0 / 2.0, 1.0 / 2.0, 1.0]], dtype=np.float32)

            # Align the final image using the homography matrix
            aligned_ch2 = cv2.warpPerspective(ch2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # Store aligned channel
            aligned_channels.append(aligned_ch2)

        # Merge aligned channels back into an RGB image
        aligned_image = cv2.merge(aligned_channels)
    else:
        raise ValueError("Both images must be RGB.")

    # Save the aligned image with metadata
    save_image(img_save_name, aligned_image, metadata2)

def process_multispec_set(green_image_path, output_directory):
    """Process one set of multispectral images (G, R, RE, NIR); using green (G) as reference for the other channels."""
    base_name = os.path.basename(green_image_path)    
    # Load the green band image (reference image)
    print("Loading image " + green_image_path)
    imageG, metadataG = load_image(green_image_path)
    copy_file(green_image_path, os.path.join(output_directory, os.path.basename(green_image_path)))  # just copy the green band image to the output directory
    
    # Process each spectral band
    for band in ['V']:
        band_image_path = green_image_path.replace('_G.TIF', f'_{band}.TIF')
        if os.path.exists(band_image_path):
            # Load the current band image
            image_band, metadata_band = load_image(band_image_path)
            
            # Align the current band image to the green band image
            img_save_name = os.path.join(output_directory, f'{base_name.replace("_G.TIF", f"_{band}.TIF")}')
            align_images_rgb(imageG, image_band, metadata_band, img_save_name)



def process_directory(input_dir):
    """Process an entire directory with multiple sets of multispectral images"""
    # Create the output directory if it doesn't exist
    input_dir = os.path.abspath(input_dir)
    output_directory = os.path.join(input_dir, 'aligned')
    os.makedirs(output_directory, exist_ok=True)
    print("Output-directory ok")
    
    # Find all green band images (reference images)
    green_images = glob(os.path.join(input_dir, '*_G.TIF'))         # 731_0041_MS_RE
    print("Green-channel images found. Starting multi-threading.")
    # Process sets of 4 multispectral images (4 spectral bands) in paralell
    with Pool(n_threads) as pool:
        pool.starmap(process_multispec_set, zip(green_images, repeat(output_directory)))
        pool.close()
        pool.join() # wait for worker processes to exit
    
    print("All images successfully processed.")


if __name__ == "__main__":
    input_dir = sys.argv[1]
    n_threads = int(sys.argv[2])
    start_time = time.time()
    process_directory(input_dir)
    end_time = time.time()
    print("Time elapsed with settings:" + str(number_of_iterations) + " and " + str(termination_eps) + " : " +  str(end_time - start_time))
