from ultralytics import YOLO

# Load the YOLOv8 model — start from scratch or fine-tune from pretrained
model = YOLO('yolov8n.pt')  

model.names[0] = "worker"
# Train the model with multi-scale and augmentations
if __name__ == "__main__":
    model.train(
        data='data.yaml',
        name = '1_class_nano_1280_5000(best)',
        pretrained=True,
        epochs=200,
        imgsz=1280,                 # Large image size helps with small object detection
        batch=4,
        multi_scale=False,
        mosaic=1.0,              
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4, 
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mixup=0.2,
        box=7.5,
        cls=1.5,
        dfl=1.0,
        close_mosaic=15,
        optimizer='SGD',
        workers=4,
        patience=30,
        val=True,
        device = 0 
        )

# Evaluate the model on the validation set and print metrics
    metrics = model.val()  # You can also pass specific dataset: model.val(data='dataset/data.yaml')

    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")



