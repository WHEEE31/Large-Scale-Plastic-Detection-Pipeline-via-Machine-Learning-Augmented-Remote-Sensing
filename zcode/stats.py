import os
import pandas as pd
import numpy as np
from PIL import Image
import csv

ROOT = "C:\\Users\\super\\Documents\\projects\\Yayscires"

def analyze_pixel_intensity(image_paths, labels_paths, root_path, band_names):
    """
    Analyze pixel intensity distributions for plastics versus non-plastics across datasets.

    Args:
        image_paths (dict): A dictionary mapping band names to file paths for one image per band.
        labels_paths (dict): A dictionary mapping band names to corresponding CSV label file paths.
        root_path (str): Root directory containing image and label subdirectories.
        band_names (list): List of band names (e.g., ['NIR', 'R', 'RE', 'G', 'RGB']).

    Returns:
        dict: Running averages of mean pixel intensities for plastics and non-plastics per band.
    """
    running_avg = {band: {'plastic': [], 'non_plastic': []} for band in band_names}

    for band in band_names:
        # Open the image
        image_path = image_paths[band]
        img = Image.open(os.path.join(root_path, f'chips_{band}', image_path)).convert('L')
        img_array = np.array(img)

        # Load the corresponding labels (assuming CSV has bounding boxes for plastics)
        labels_path = labels_paths[band]
        labels = pd.read_csv(os.path.join(root_path, labels_path))

        # Filter the labels for the current image
        labels = labels[labels['filename'] == image_path]

        # Create a binary mask for plastics (initially all False)
        plastic_mask = np.zeros(img_array.shape[:2], dtype=bool)

        # Mark pixels as plastic based on bounding boxes for this image
        for _, row in labels.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Ensure that bounding box is within image limits (no negative values)
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, img_array.shape[1])
            ymax = min(ymax, img_array.shape[0])

            # Update the mask for this bounding box (mark as True inside the box)
            plastic_mask[ymin:ymax, xmin:xmax] = True
        """
        # Debugging the mask: print how many pixels are marked as plastic
        print(f"Plastic Mask (Sum of Trues): {np.sum(plastic_mask)}")
        print(f"Mask Shape: {plastic_mask.shape}")
        print(f"Image Shape: {img_array.shape}")
        """
        plastic_region = img_array.copy()
        plastic_region[~plastic_mask] = 0  # Set non-plastic regions to black for visualization
        #Image.fromarray(plastic_region).show()

        # Now separate the plastic and non-plastic pixels
        plastic_pixels = img_array[plastic_mask]
        non_plastic_pixels = img_array[~plastic_mask]
        """
        # Debugging the pixel count
        print(f"Plastic Pixels Count: {len(plastic_pixels)}")
        print(f"Non-Plastic Pixels Count: {len(non_plastic_pixels)}")
        """
        # Calculate mean pixel intensities for plastics and non-plastics
        plastic_mean = np.mean(plastic_pixels) if len(plastic_pixels) > 0 else 0
        non_plastic_mean = np.mean(non_plastic_pixels) if len(non_plastic_pixels) > 0 else 0

        # Update running averages
        running_avg[band]['plastic'].append(plastic_mean)
        running_avg[band]['non_plastic'].append(non_plastic_mean)
        """
        # Print summary for the current band
        print(f"Band: {band}")
        print(f"Mean Intensity (Plastic): {plastic_mean}")
        print(f"Mean Intensity (Non-Plastic): {non_plastic_mean}")
        print('-' * 40)
        """

    # Compute overall running averages across all images
    for band in band_names:
        running_avg[band]['plastic'] = np.mean(running_avg[band]['plastic'])
        running_avg[band]['non_plastic'] = np.mean(running_avg[band]['non_plastic'])

    return running_avg

def extract_unique_chips_from_csv(csv_path):
    """Extract unique chip numbers from the 'filename' column of a CSV file."""
    chips_set = set()  # Use a set to avoid duplicates
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row['filename']  # Assuming the column name is 'filename'
            # Extract the chip number using the format "chip_{band}{chipnum}.tif"
            parts = filename.split('_')  # Split at underscores
            if len(parts) > 1 and parts[0] == 'chip':  # Ensure the format matches
                chipnum = ''.join(filter(str.isdigit, parts[1]))  # Extract digits
                if chipnum.isdigit():  # Ensure it's a valid number
                    chips_set.add(int(chipnum))
    return sorted(chips_set)  # Return as a sorted list

if __name__ == "__main__":
    # Define file paths for one image per band and corresponding labels
    root_path = os.path.join(ROOT, 'new_data', 'chips')
    band_names = ['G', 'NIR', 'R', 'RE', 'RGB']
    plastic = [0 for x in band_names]
    non_plastic = [0 for x in band_names]
    
    chips = extract_unique_chips_from_csv(os.path.join(root_path, 'labels_G.csv'))

    for l, chip_num in enumerate(chips):
        print(f'processing {l+1} / {len(chips)}', end = '\r')
        image_paths = {band : f'chip_{band.lower()}{chip_num}.{'jpg' if band == 'RGB' else 'tif'}' for band in band_names}
        labels_paths = {band : f'labels_{band}.csv' for band in band_names}

        # Analyze pixel intensities
        running_avg = analyze_pixel_intensity(image_paths, labels_paths, root_path, band_names)
        # Print final running averages
        for i, band in enumerate(band_names):
            plastic[i] = plastic[i] + running_avg[band]['plastic']
            non_plastic[i] = non_plastic[i] + running_avg[band]['non_plastic']
    for i in range(len(band_names)):
        plastic[i] = plastic[i] / len(chips)
        non_plastic[i] = non_plastic[i] / len(chips)
        print(f"{band_names[i]} - Plastic: {plastic[i]:.2f}, Non-Plastic: {non_plastic[i]:.2f}, Ratio: {plastic[i]/non_plastic[i]:.2f}")
