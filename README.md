# Semi-Supervised-Labelling-Model-for-Ailytics
This project focuses on developing a decent-enough object detection model to carry out semi-supervised labelling, which is the generation of an initial first pass of annotations in Labelme JSON file format on an unseen batch of images. This lays the groundwork for data labelling, accelerating and improving efficiency of the manual process. The model utilises transfer learning from YOLOv8 pretrained weights trained on a dataset of 5000 images split into an 80/10/10 train/val/test ratio. By finetuning the pretrained models and adjusting the hyperameters for maximum mAP50, mAP50-95 and recall, time taken for data labelling can be reduced. 

Note: In view of confidentiality policy of the company, images from the dataset used and model weights have been removed from this repository.

However some limitation of this project includes the computing power(the models were trained on an ACER NITRO V15 laptop with NVIDIA GeForce RTX 2050 and Cuda version: 12.3) and limited access to company's data.
## Results

