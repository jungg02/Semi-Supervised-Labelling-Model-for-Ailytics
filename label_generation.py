import os
import json
from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8n.pt") 

def run_yolo_and_export_json(image_path, output_json_dir):
    # Run inference
    results = model(image_path)
    detections = results[0]

    image_name = os.path.basename(image_path)
    image_id = os.path.splitext(image_name)[0]

    shapes = []
    if detections.boxes.data is not None:
        for box in detections.boxes.data:
            x1, y1, x2, y2, conf, class_id = box.tolist()
            class_id = int(class_id)
            class_name = model.names[class_id]
            shape = {
                "label": class_name,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)

    if not shapes:
        print(f"No detections in: {image_name}")
        return

    json_output = {
        "version": "5.7.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None
    }


    # Ensure output directory exists
    os.makedirs(output_json_dir, exist_ok=True)

    json_filename = f"{image_id}.json"
    json_path = os.path.join(output_json_dir, json_filename)

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)

    print(f"Saved: {json_path}")


input_folder = r"C:\Users\ailyt\Desktop\side project\semi-supervised labelling\images\test"                     # Replace with your image path
output_folder = r"C:\Users\ailyt\Desktop\side project\semi-supervised labelling\images\results"                # Folder where JSONs will be saved

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        run_yolo_and_export_json(image_path, output_folder)