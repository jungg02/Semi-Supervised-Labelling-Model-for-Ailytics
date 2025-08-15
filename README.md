# Semi-Supervised-Labelling-Model-for-Ailytics
This project focuses on developing a decent-enough object detection model to carry out semi-supervised labelling, which is the generation of an initial first pass of annotations in Labelme JSON file format on an unseen batch of images. This lays the groundwork for data labelling, accelerating and improving efficiency of the manual process. The model utilises transfer learning from YOLOv8 pretrained weights trained on a dataset of 5000 images split into an 80/10/10 train/val/test ratio.

The project also explores different combinations of hyperparameters to achieve the best performance in a resource constraint environment. Models were trained on an ACER NITRO V15 laptop with NVIDIA GeForce RTX 2050 (4BG VRAM), Cuda version: 12.3 and size of my dataset was limited to 5000 images.

Note: In view of confidentiality policy of the company, images from the dataset used and model weights have been removed from this repository.

## Results
Models were trained to perform object detection on 3 classes: worker, vests and hardhats. Per-class metrics and performance curves can be found in the respective model folders. 
| Model  | imgsz | Precision | Recall | mAP50 | mAP50-95 |
| -------- | ------- | ------- | ------- | ------- | ------- |
| YOLOv8n  | 1280 | 0.804 | 0.641 | 0.723 | 0.414 |
| YOLOv8s |  960 | 0.816 | 0.612 | 0.700 | 0.415 |
* imgsz had to be reduced to 960 for the YOLOv8s model due to resource constraint

In order to prevent CUDA Out-of-Memory(OOM) errors while maintaining a reasonable training time, only small and nano sizes of the YOLO architecture were considered for this project. 

Larger image resolutions(`imgsz=960` and `imgsz=1280`) were also chosen due to the nature of the dataset used(ie. images taken from both faraway and closeup cameras in construction sites and other heavy industries) so as to prevent the loss of finer details and smaller classes, such as hardhats, from disappearing. 

Multiscale was also turned off as a separate experiment was conducted on the same dataset showing minor improvements in performance metrics while increasing training times by 2-3x.

** Evaluation
This experiment had showed that when it comes to developing a model for generating annotations, using a smaller model at a higher image size may prove to be more beneficial, allowing the model to create more bounding boxes while keeping false negatives low. This helps to reduce the time taken the data labelling step in the data pipeline. 


