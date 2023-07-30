# Image-Segmentation-UNet
**UNetify:** Masterful Image Segmentation with Pre-Trained UNet using Pytorch

This project lays the groundwork for creating highly advanced applications, such as segmenting images into distinct objects or persons—technology often utilized in self-driving cars

![image](https://github.com/DurgaSandeep25/Image-Segmentation-UNet/assets/38128597/356c3cc7-23ef-4898-b4e8-e8e1e6013ca3)


**Objective:** To identify all individuals in an image accuracy. No more mystery faces! This can be used for crowd analysis, event management, or any project needing precise human detection.

Images can be as simple as following one "Single Person"
![image](https://github.com/DurgaSandeep25/Image-Segmentation-UNet/assets/38128597/2d1038c8-f0bc-4b14-9e7b-e4aa440032ba)

Or It can be as complicated as below
![image](https://github.com/DurgaSandeep25/Image-Segmentation-UNet/assets/38128597/10a0ea55-d5d7-464e-bc44-6480691a90ca)

**Data Augumentation**: Transform your data using Albumentations library (very good for augumentation in Computer Vision). Following is an example for vertical flip of an exisiting image

![image](https://github.com/DurgaSandeep25/Image-Segmentation-UNet/assets/38128597/fa70ad86-12f9-430f-acf2-ff0ae3bd5af8)

**Model Architecture** : UNet Pretrained with
Encoder = TIMM Efficient Net
Weights = Imagenet
loss function = DiceLoss + Binary Cross Entropy Loss

![image](https://github.com/DurgaSandeep25/Image-Segmentation-UNet/assets/38128597/826080dc-9d27-40a5-8d57-7b144ec04198)

**Model Performance**: As you can see, loss is not smooth, mainly because of less data points in train and validation set.


**Inference**



