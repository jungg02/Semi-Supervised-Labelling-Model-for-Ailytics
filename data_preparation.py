"" this file randomizes the dataset to prevent overfitting to certain conditions (lighting, angles, camera distances, backgrounds) and splits the data into training, val and test sets according to the specified split ratio. It also convverts the Labelme JSON files into YOLO text format to be used in model training ""

import os
import json
import random
import shutil

# CONFIGURATION
input_dir = r"C:\Users\ailyt\Desktop\side project\semi-supervised labelling\dataset\images"  # folder with images + .json files
output_dir = r"C:\Users\ailyt\Desktop\side project\semi-supervised labelling\dataset"
split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
labelme_classes = ['worker']  # adjust based on your labels

def convert_labelme_to_yolo(json_file, output_txt, class_list, img_w, img_h):
    with open(json_file, 'r') as f:
        data = json.load(f)
        for shape in data.get("shapes", []):
            # Rename 'longvest' to 'vest'
            if shape.get("label", "") == "longvest":
                shape["label"] = "vest"


    with open(output_txt, 'w') as out:
        for shape in data.get('shapes', []):
            label = shape['label']
            if label not in class_list:
                continue
            class_id = class_list.index(label)
            points = shape['points']
            x_min = min(p[0] for p in points)
            x_max = max(p[0] for p in points)
            y_min = min(p[1] for p in points)
            y_max = max(p[1] for p in points)

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def prepare_dataset():
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    random.shuffle(files)

    n = len(files)
    split_points = {
        "train": files[:int(n * split_ratio["train"])],
        "val": files[int(n * split_ratio["train"]):int(n * (split_ratio["train"] + split_ratio["val"]))],
        "test": files[int(n * (split_ratio["train"] + split_ratio["val"])):]
    }

    for split, json_list in split_points.items():
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, sub, split), exist_ok=True)

        for json_file in json_list:
            base = os.path.splitext(json_file)[0]
            img_file = base + ".jpg"  # or .png depending on your data
            json_path = os.path.join(input_dir, json_file)
            img_path = os.path.join(input_dir, img_file)

            if not os.path.exists(img_path):
                print(f"Skipping {img_file} (missing)")
                continue

            # Copy image
            shutil.copy(img_path, os.path.join(output_dir, 'images', split, img_file))

            # Load image size from JSON or actual image
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            img_w = data.get("imageWidth")
            img_h = data.get("imageHeight")

            if img_w is None or img_h is None:
                print(f"Skipping {json_file}: missing imageWidth/imageHeight")
                continue

            # Convert annotation
            yolo_txt = os.path.join(output_dir, 'labels', split, base + '.txt')
            convert_labelme_to_yolo(json_path, yolo_txt, labelme_classes, img_w, img_h)

prepare_dataset()

