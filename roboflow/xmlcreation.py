import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image
import tqdm

csv_path = "new_data/chips/labels_G.csv"
images_dir = "roboflow/images"
output_dir = "roboflow/labels"
os.makedirs(output_dir, exist_ok=True)

# Load all rows grouped by filename
labels = {}
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in tqdm.tqdm(reader):
        fname = os.path.splitext(row['filename'])[0] + ".jpg"  # ensure .jpg extension
        box = {
            'xmin': int(float(row['xmin'])),
            'ymin': int(float(row['ymin'])),
            'xmax': int(float(row['xmax'])),
            'ymax': int(float(row['ymax'])),
            'class': row['class']
        }
        labels.setdefault(fname, []).append(box)

# Generate XML for each image
for fname, objects in tqdm.tqdm(labels.items()):
    img_path = os.path.join(images_dir, fname)
    if not os.path.exists(img_path):
        print(f"Warning: image not found for {fname}")
        continue

    with Image.open(img_path) as img:
        width, height = img.size

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = os.path.basename(images_dir)
    ET.SubElement(annotation, "filename").text = fname
    ET.SubElement(annotation, "path").text = os.path.abspath(img_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj['class']
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"

        bbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj['xmin'])
        ET.SubElement(bbox, "ymin").text = str(obj['ymin'])
        ET.SubElement(bbox, "xmax").text = str(obj['xmax'])
        ET.SubElement(bbox, "ymax").text = str(obj['ymax'])

    xml_str = ET.tostring(annotation, encoding="utf-8")
    xml_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".xml")

    with open(xml_path, "wb") as f:
        f.write(xml_str)
