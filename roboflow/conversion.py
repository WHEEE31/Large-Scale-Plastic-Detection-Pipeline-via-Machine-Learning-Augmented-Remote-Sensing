from PIL import Image
import os
import tqdm

input_dir = "new_data/chips/chips_G"    # folder containing .tif files
output_dir = "roboflow/images"   # folder to save .jpg files
os.makedirs(output_dir, exist_ok=True)

for filename in tqdm.tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(".tif"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_dir, output_filename)

        with Image.open(input_path) as img:
            img = img.convert("RGB")  # ensures compatibility
            img.save(output_path, "JPEG", quality=95)

