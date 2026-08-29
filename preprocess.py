import os
import glob
import random
import time
import pickle
import csv
import rawpy
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance, ImageDraw
from sklearn.model_selection import train_test_split

DIRS = []

def set_consts(dirs):
    global DIRS
    DIRS = dirs
IN = 134

def test_preprocessing():
    in_df = pd.read_csv(os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], 'labels_G.csv'))
    
    for idx in range(1162):
        print(in_df.iloc[idx])
        image = Image.open(os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], 'chips_G', in_df.iloc[idx,0])).copy()
        img = Img(image)
        boxes = in_df.iloc[idx, 1:5].values.astype(float).reshape(-1, 4)
        for box in boxes:
            img.add_bounding_box((box[0], box[1], box[2], box[3]), 'hi')
        img.show_bounding_boxes()
        time.sleep(2)
    '''
    data = pd.read_csv(os.path.join(DIRS[0], DIRS[3][0], 'labeled', 'annotations.csv'))
    name = 'g_1.tif'           # which became entry 16, chip_g33.tif
    image = Image.open(os.path.join(DIRS[0], DIRS[3][0], 'raw_data', name))
    img = Img(image)
    for _, row in data[data['filename'] == name].iterrows():
        img.add_bounding_box(row[4:8], row[3])
    #img.show_bounding_boxes()

    coords = [3,1]
    img.crop([coords[0]*640, coords[1]*640, coords[0]*640+640, coords[1]*640+640])
    
    img.show_bounding_boxes()
    '''

class Img:
    def __init__(self, pil_image, boxes=None, testing=False):
        """
        Initialize the custom Image class.
        :param pil_image: A PIL.Image object.
        :param boxes: Optional; a list of bounding boxes as (x_min, y_min, x_max, y_max, class_name).
        """
        self.testing = testing
        self.image = pil_image
        self.buffer = 50
        self.bounding_boxes = boxes if boxes else []  # List of bounding boxes (with class names).

    def add_bounding_box(self, bbox, class_name):
        """Add a bounding box with a classification name."""
        self.bounding_boxes.append((*bbox, class_name))

    def get_bounding_boxes(self):
        """Return the list of bounding boxes with class names."""
        new_boxes = []
        for bbox in self.bounding_boxes:
            new_bbox = [int(bbox[x]) for x in range(4)]+[bbox[4]]
            new_boxes.append(new_bbox)
        return new_boxes   

    def show_bounding_boxes(self):
        temp = self.image.copy()
        draw = ImageDraw.Draw(temp)
        for box in self.get_bounding_boxes():
            draw.rectangle((box[0],box[1],box[2],box[3]),outline="blue",width=4)
        temp.show()

    @property
    def size(self):
        """Return the size of the image."""
        return self.image.size   

    def save(self, path, format=None):
        """Save the wrapped PIL image."""
        self.image.save(path, format=format)

    def copy(self):
        """Create a copy of the image and its bounding boxes."""
        return Img(self.image.copy(), self.bounding_boxes[:])
    
    def show(self):
        self.image.show()
    
    # transformations
    def rotate(self, angle, resample=0, expand=False, center=None, translate=None, fillcolor=None):
        """Rotate the image and update bounding boxes."""
        old_width, old_height = self.image.size
        self.image = self.image.rotate(angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor)
        new_width, new_height = self.image.size
        rad_angle = np.deg2rad(-angle)
        cx, cy = old_width / 2, old_height / 2

        if expand:
            new_cx, new_cy = new_width / 2, new_height / 2
        else:
            new_cx, new_cy = cx, cy
        rotation_matrix = np.array([
            [np.cos(rad_angle), -np.sin(rad_angle)],
            [np.sin(rad_angle),  np.cos(rad_angle)]
        ])

        updated_bboxes = []
        for bbox in self.bounding_boxes:
            corners = np.array([
                [bbox[0], bbox[1]],  # Top-left
                [bbox[2], bbox[1]],  # Top-right
                [bbox[0], bbox[3]],  # Bottom-left
                [bbox[2], bbox[3]]   # Bottom-right
            ])

            translated_corners = corners - np.array([cx, cy])
            rotated_corners = np.dot(translated_corners, rotation_matrix.T)
            final_corners = rotated_corners + np.array([new_cx, new_cy])

            x_coords = final_corners[:, 0]
            y_coords = final_corners[:, 1]
            new_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords), bbox[4])
            updated_bboxes.append(new_bbox)
        
        self.bounding_boxes = updated_bboxes
        if self.testing:
            self.show_bounding_boxes()

    def resize(self, size, resample=0, box=None, reducing_gap=None):
        """Override resize to also update bounding boxes."""
        old_width, old_height = self.size
        self.image = self.image.resize(size, resample=resample, box=box, reducing_gap=reducing_gap)
        width_scale = size[0] / old_width
        height_scale = size[1] / old_height
        updated_bboxes = []
        for bbox in self.bounding_boxes:
            new_bbox = (
                int(bbox[0] * width_scale),
                int(bbox[1] * height_scale),
                int(bbox[2] * width_scale),
                int(bbox[3] * height_scale),
                bbox[4]  # Keep the class name
            )
            # Ensure xmin < xmax and ymin < ymax
            new_bbox = (min(new_bbox[0], new_bbox[2]), min(new_bbox[1], new_bbox[3]), 
                        max(new_bbox[0], new_bbox[2]), max(new_bbox[1], new_bbox[3]), new_bbox[4])
            updated_bboxes.append(new_bbox)
        self.bounding_boxes = updated_bboxes
        if self.testing():
            self.show_bounding_boxes()

    def transpose(self, method):
        """Override transpose to also update bounding boxes."""
        original_width, original_height = self.size
        self.image = self.image.transpose(method)
        if method == Image.FLIP_LEFT_RIGHT:
            updated_bboxes = [
                (original_width - bbox[2], bbox[1], original_width - bbox[0], bbox[3], bbox[4])
                for bbox in self.bounding_boxes
            ]
        elif method == Image.FLIP_TOP_BOTTOM:
            updated_bboxes = [
                (bbox[0], original_height - bbox[3], bbox[2], original_height - bbox[1], bbox[4])
                for bbox in self.bounding_boxes
            ]
        else:
            updated_bboxes = self.bounding_boxes

        # Ensure xmin < xmax and ymin < ymax
        updated_bboxes = [
            (min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3]), bbox[4])
            for bbox in updated_bboxes
        ]
        self.bounding_boxes = updated_bboxes
        if self.testing:
            self.show_bounding_boxes()

    def crop(self, box):
        """
        Crop the image and update the bounding boxes accordingly.
        Box format: [left, upper, right, lower]
        """
        # Perform the crop operation on the image
        cropped_image = self.image.crop(box)

        # Adjust bounding boxes for the cropped region
        cropped_bboxes = []
        for bbox in self.bounding_boxes:
            coords = [bbox[0] - box[0], bbox[1] - box[1], bbox[2] - box[0], bbox[3] - box[1]]
            old = coords[:]
            length = box[2]-box[0]
            height = box[3]-box[1]

            fit = [i for i in range(4) if coords[i] >= (0+self.buffer) and coords[i] <= ([length, height][i%2]-self.buffer)]
            if not (len(fit) < 2 or (len(fit) == 2 and fit[1]-fit[0] == 2)):  # some point is within the area
                unfit = [i for i in range(4) if i not in fit and (coords[i] < 0 or coords[i] > [length, height][i%2])]
                for i in unfit:
                    a,b = abs(coords[i] - 0),0
                    if i%2 == 0:
                        b = abs(coords[i] - length)
                    else:
                        b = abs(coords[i] - height)                  
                    coords[i] = {a: 0, b: length}[min(a,b)]
                new_bbox = coords[:]
                new_bbox.append(bbox[4])
                cropped_bboxes.append(new_bbox)
        
        # Update the image and bounding boxes for the cropped region
        self.image = cropped_image
        self.bounding_boxes = cropped_bboxes
        if self.testing:
            self.show_bounding_boxes()
        return self

    def change_brightness(self, factor):
        """Adjust the brightness of the image."""
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(factor)

    def change_contrast(self, factor):
        """Adjust the contrast of the image."""
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(factor)

class JitterUtil:
    def __init__(self, csv_input_path, csv_output_path):
        self.algorithms = []
        self.INcsv = pd.read_csv(csv_input_path)  # Reading CSV for bounding box data
        self.OUTcsv = csv_output_path  # Output CSV file path

        # Define the jitter transformations and their probability ranges
        self.jitters = [("flip", 50, 100),
                        ("rotate", 30, 60, 100)]

    def clear(self):
        self.algorithms = []

    def jitter(self, numTransforms):
        """Generate a set of random transformations"""
        for k in range(numTransforms):
            steps = []
            for i in range(10):
                step = str(random.randint(0, len(self.jitters)-1))
                step += "." + str(random.randint(0, 100))
                steps.append(step)
            self.algorithms.append(steps)

    def generate(self, images, steps=None):
        """Generate jittered images based on random steps"""
        if not steps:
            steps = self.algorithms

        i = 0
        for image in images:
            # Apply all jittering steps
            for step in steps[i]:
                step = [int(x) for x in step.split('.')]
                if step[0] == 0:  # flip
                    if step[1] < self.jitters[0][1]:  # Flip top-bottom
                        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    elif step[1] < self.jitters[0][2]:  # Flip left-right
                        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                elif step[0] == 1:  # rotate
                    if step[1] < self.jitters[1][1]:  # 90 degrees
                        image.rotate(90)
                    elif step[1] < self.jitters[1][2]:  # 180 degrees
                        image.rotate(180)
                    elif step[1] < self.jitters[1][3]:  # 270 degrees
                        image.rotate(270)
                elif step[0] == 2:  # brightness
                    image.change_brightness(random.uniform(0.8,1.2))
                elif step[0] == 3:  # contrast
                    image.change_contrast(random.uniform(0.8,1.2))

            # Resize image to 640x640
            #image.resize((640, 640))

            i += 1

def jitter():
    print("Starting jittering...")
    outCSV = []
    inCSV = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][2][0], 'annotations.csv')
    root = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0])
    outCSV.append(os.path.join(root, 'labels_G.csv'))
    outCSV.append(os.path.join(root, 'labels_NIR.csv'))
    outCSV.append(os.path.join(root, 'labels_R.csv'))
    outCSV.append(os.path.join(root, 'labels_RE.csv'))
    outCSV.append(os.path.join(root, 'labels_RGB.csv'))
    paths = []  # g, nir, r, re, rgb #
    root = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][3][0])
    paths.append([os.path.join(root, f) for f in glob.glob(os.path.join(root, 'g_*'))])
    paths.append([os.path.join(root, f) for f in glob.glob(os.path.join(root, 'nir_*'))])
    paths.append([os.path.join(root, f) for f in glob.glob(os.path.join(root, 'r_*'))])
    paths.append([os.path.join(root, f) for f in glob.glob(os.path.join(root, 're_*'))])
    paths.append([os.path.join(root, f) for f in glob.glob(os.path.join(root, 'rgb_*'))])
    
    # Read the input CSV into a DataFrame
    in_df = pd.read_csv(inCSV)
    
    # Initialize the output CSV file (just headers for now)
    out_df = []
    out_df.append(pd.read_csv(outCSV[0]))
    out_df.append(pd.read_csv(outCSV[1]))
    out_df.append(pd.read_csv(outCSV[2]))
    out_df.append(pd.read_csv(outCSV[3]))
    out_df.append(pd.read_csv(outCSV[4]))
    
    j = JitterUtil(inCSV, outCSV)
    
    totals = [0,0]
    for i in range(IN):
        chips = []
        
        for k in range(len(paths)):  # Loop through each band
            image_path = paths[k][i]
            image = Image.open(image_path).copy()
            img = Img(image)  # Convert to the custom Image class
            
            # Get bounding boxes for the current image from the CSV
            image_name = os.path.basename(image_path)
            image_data = in_df[in_df['filename'] == image_name]
            
            for _, row in image_data.iterrows():
                # Add bounding boxes from inCSV to the Image object
                img.add_bounding_box((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])), row['class'])
            
            # Crop image into smaller chips (e.g., 12 or 16 chips)
            width, height = 2592, 1944
            chip_size = 640  # Set the size of each chip (adjustable)
            overlap = 0
            
            cropped_chips = []
            for x in range(0, width, chip_size-overlap):
                for y in range(0, height, chip_size-overlap):
                    # Define the cropping box
                    box = [x, y, x+chip_size, y+chip_size]
                    if box[2] > width:
                        box[0] = width - chip_size
                        box[2] = width
                    if box[3] > height:
                        box[1] = height - chip_size
                        box[3] = height
                    cropped_chip = img.copy()
                    cropped_chip.crop(box)
                    cropped_chips.append(cropped_chip)
            chips.append(cropped_chips)

        # Jittering on the cropped chips
        j.jitter(len(chips[0]))
        for l in range(len(chips)):
            j.generate(chips[l])
        j.clear()

        chip_bboxes = [[],[],[],[],[]]
        numChips = 0
        numBoxes = 0
        # Save generated chips
        for k in range(len(chips)):
            folder_path = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], f'chips_{["G", "NIR", "R", "RE", "RGB"][k]}')
            for l in range(len(chips[k])):
                chip = chips[k][l]
                name = f'chip_{["g", "nir", "r", "re", "rgb"][k]}{l+i*len(chips[k])}.{"tif" if k != 4 else "jpg"}'
                #time.sleep(0.01)
                boxes = chip.get_bounding_boxes()
                if len(boxes) != 0:
                    numChips += 1
                    numBoxes += len(boxes)
                    chip.save(os.path.join(folder_path, name))
                    for bbox in boxes:
                        chip_bboxes[k].append({
                            'filename': name,
                            'xmin': bbox[0],
                            'ymin': bbox[1],
                            'xmax': bbox[2],
                            'ymax': bbox[3],
                            'class': bbox[4]
                        })
        
        for k in range(5):
            for bbox in chip_bboxes[k]:
                out_df[k] = pd.concat([out_df[k], pd.DataFrame([bbox])], ignore_index=True)
            out_df[k].to_csv(outCSV[k], index=False)

        totals[0] += numChips
        totals[1] += numBoxes
        print(f"saved {numChips} chips with {numBoxes} bounding boxes from image #{i+1} / {IN}             ", end = "\r")

    # Save the output CSV after processing all images
    for k in range(5):
        out_df[k].to_csv(outCSV[k], index=False)

    print(f"finished jittering - generated {totals[0]} chips with {totals[1]} bounding boxes   ({int(totals[1] / 5)} training entities per dataset)")

def prep():
    print("Prepping train/test CSV and stats.txt files")
    
    root = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0])
    save = os.path.join(DIRS[0], 'evaluate')
    np.random.seed(42)

    for band in ['G', 'NIR', 'R', 'RE', 'RGB']:
        print(f"Processing band {band}...")
        df = pd.read_csv(os.path.join(root, f"labels_{band}.csv"))
        unique_filenames = df['filename'].unique()

        np.random.shuffle(unique_filenames)
        total_boxes = len(df)
        train_box_limit = int(total_boxes * 0.7)
        train_filenames, test_filenames = [], []
        train_box_count = 0

        for filename in unique_filenames:
            box_count = len(df[df['filename'] == filename])
            if train_box_count + box_count <= train_box_limit:
                train_filenames.append(filename)
                train_box_count += box_count
            else:
                test_filenames.append(filename)

        train_dataset = df[df['filename'].isin(train_filenames)]
        test_dataset = df[df['filename'].isin(test_filenames)]
        train_dataset.to_csv(os.path.join(save, f'train_{band}.csv'), index=False)
        test_dataset.to_csv(os.path.join(save, f'test_{band}.csv'), index=False)

        # Output stats
        print(f"Band {band}:")
        print(f"  Training set: {len(train_filenames)} images, {len(train_dataset)} boxes")
        print(f"  Testing set: {len(test_filenames)} images, {len(test_dataset)} boxes")

def rename():
    sn = 0
    targetLength = 1000
    for dng_path in glob.glob('new_data/raw_data/aligned/*.DNG'):
        print(sn)
        # Open the .DNG file with rawpy
        with rawpy.imread(DIRS[0] + "\\" + dng_path) as raw:
            # Convert to an RGB image
            rgb_image = raw.postprocess()

        # Convert to Pillow image
        image = Image.fromarray(rgb_image)

        # Save as .png in the output directory
        image.save(os.path.join(DIRS[0],'new_data/raw_data', "rgb_" + str(sn)+".jpg"))

        sn += 1

    sn=0
    for f in glob.glob('new_data/raw_data/aligned/*.TIF'):
        print(sn, end="\r")
        image = Image.open(DIRS[0] + "\\" + f).copy()
        image.save(os.path.join(DIRS[0],'new_data/raw_data', ["g","nir","r","re"][sn%4] + "_" + str(sn//4)+".tif"))
        sn += 1
    
    for f in range(134):
        print(sn, end='\r')
        image = Image.open(DIRS[0] + "\\" + 'new_data/raw_data/rgb_' + str(f) + '.jpg').copy()
        image.save(os.path.join(DIRS[0], 'new_data/test', str(sn) + "_V.TIF"), format = "tiff")
        sn += 1
    sn = 0
    for f in range(134):
        print(sn, end='\r')
        image = Image.open(DIRS[0] + "\\" + 'new_data/raw_data/g_' + str(f) + '.tif').copy()

        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = (array - array.min()) / (array.max() - array.min()) * 255
            array = array.astype(np.uint8)
        image = Image.fromarray(array, mode='L')
        array = np.stack((array, array, array), axis=-1)

        image = Image.fromarray(array, mode = "RGB")
        image.save(os.path.join(DIRS[0], 'new_data/test', str(sn) + "_G.TIF"), format = "tiff")
        sn += 1

# parsing labelIMG annotations to CSV
def parse_xml_to_csv(xml_folder):
    rows = []
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            for member in root.findall('object'):
                row = {
                    'filename': root.find('filename').text,
                    'width': int(root.find('size/width').text),
                    'height': int(root.find('size/height').text),
                    'class': member.find('name').text,
                    'xmin': int(member.find('bndbox/xmin').text),
                    'ymin': int(member.find('bndbox/ymin').text),
                    'xmax': int(member.find('bndbox/xmax').text),
                    'ymax': int(member.find('bndbox/ymax').text)
                }
                rows.append(row)
    return pd.DataFrame(rows)

def makeCSV():
    xml_folder = os.path.join(DIRS[0],'new_data/labeled/labeled_G/')

    # Convert XML to DataFrame and save as CSV
    output_csv = "annotations_G.csv"
    df = parse_xml_to_csv(xml_folder)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved as {output_csv}")

def cropRGB():    
    input_dir = os.path.join(DIRS[0], 'new_data/raw_data/')
    output_dir = os.path.join(DIRS[0], 'new_data/test/') 

    
    scale_x_adjust = 0.974              # 0.974
    scale_y_adjust = 0.965              # 0.965
    crop_x_adjust = 0
    crop_y_adjust = 0
    degrees = -0.6

    original_size = (5280, 3956)
    target_size = (2592, 1944)
    center1 = (1328,803)    # small     # (1328,803)
    center2 = (2709,1635)   # big       # (2709,1635)
    point1 = (1414,895)                 # (1414,895)
    point2 = (2855,1795)                # (2855,1795)

    scale_x = (point2[0] - center2[0]) / (point1[0] - center1[0]) * scale_x_adjust
    scale_y = (point2[1] - center2[1]) / (point1[1] - center1[1]) * scale_y_adjust
    (x1,y1) = ((0 - center1[0]) * scale_x + center2[0],
               (0 - center1[1]) * scale_y + center2[1])
    (x2,y2) = ((target_size[0] - center1[0]) * scale_x + center2[0],
               (target_size[1] - center1[1]) * scale_y + center2[1])

    sn=0
    for f in range(0,134):
        print(sn, end = "\r")
        image = Image.open(DIRS[0] + "\\" + 'new_data/raw_data/rgb_' + str(f) + '.jpg').copy()
        image = image.rotate(degrees)
        image = image.crop((x1+crop_x_adjust, y1+crop_y_adjust, x2+crop_x_adjust, y2+crop_y_adjust))
        image = image.resize(target_size, Image.LANCZOS)
        image.save(os.path.join(output_dir, "rgb_" + str(sn) + ".jpg"))
        sn += 1

def updateCSV():

    rows_to_add = []
    with open(os.path.join(DIRS[0], DIRS[3][0], DIRS[3][2][0], DIRS[3][2][1][0], 'annotations_G.csv'), mode='r') as file:
        reader = csv.reader(file)
        title = []
        i = 0
        for row in reader:
            i += 1
            if row[0] == 'filename':
                title = row[:]
                continue
            bandnum = 0
            for band in ['g','nir','r','re','rgb']:
                bandnum += 1
                new = row[:]
                new[0] = row[0].replace('g_',f'{band}_')
                if band == 'rgb':
                    new[0] = new[0].replace('.tif','.jpg')
                rows_to_add.append(new)
    print(title)
    with open(os.path.join(DIRS[0], DIRS[3][0], DIRS[3][2][0], 'annotations.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        rows_to_add.append(title)
        writer.writerows(rows_to_add)

def augment(image, band, scaling_factors, noise_params):

    if band != 4:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = (array - array.min()) / (array.max() - array.min()) * 255
            array = array.astype(np.uint8)
        image = Image.fromarray(array, mode='L')
        array = np.stack((array, array, array), axis=-1)

        image = Image.fromarray(array, mode="RGB")
    else:
        image = image.convert("RGB")
    
    """
    array = np.asarray(image)
    array = array / 255.0

    array = array * scaling_factors[band]
    std_dev = noise_params[band]
    noise = np.random.normal(0, std_dev, array.shape)
    array = array + noise
    array = np.clip(array, 0, 1)
    image_uint8 = (array * 255).astype(np.uint8)
    image = Image.fromarray(image_uint8).convert("RGB")
    """

    return image

def diversify(scaling_factors=0, noise_params=0, sample_size = -1):
    print("Diversifying datasets...")
    root = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0])
    if scaling_factors == 0:
        scaling_factors = {
        0: 1.0,
        1: 1.2,
        2: 1.1,
        3: 0.9,
        4: 1.0,
    }
    if noise_params == 0:
        noise_params = {
        0: 0.05,
        1: 0.1,
        2: 0.08,
        3: 0.03,
        4: 0.07,
    }
    folder_path = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0])
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, subfolder))]

    bandnum = 0
    for subfolder in subfolders:
        i = 0
        for item in os.listdir(subfolder):
            if i == sample_size:
                break
            item_path = os.path.join(subfolder, item)
            image = Image.open(item_path)
            image = augment(image, bandnum, scaling_factors, noise_params)
            image.save(os.path.join(root, f'chips_{['G','NIR','R','RE','RGB'][bandnum]}', item))
            print(f"Augmented {item}                           ", end = "\r")
            i += 1

        bandnum += 1
    print("finished diversifying!                                                   ")

def find_diversity_params(testSize):
    base_dataset = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0])

    scaling_factors = {
        0: 1.0,
        1: 1.2,
        2: 1.1,
        3: 0.9,
        4: 1.0,
    }
    noise_params = {
        0: 0.05,
        1: 0.1,
        2: 0.08,
        3: 0.03,
        4: 0.07,
    }

    options = []
    bestParams = []
    bestScore = float('-inf')

    return scaling_factors, noise_params


