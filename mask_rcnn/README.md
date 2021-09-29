
# Mask RCNN implementation to detect ice wedge polygons


Mask R-CNN is intended to perform instance segmentation,that is detecting bounding boxes and segmentation masks for each instance of an object in an image.

The code is based on https://github.com/matterport/Mask_RCNN using Python 3, Keras and TensorFlow using  a Feature Pyramid Network (FPN) and a ResNet101 backbone [(based on this paper)](https://arxiv.org/pdf/1703.06870.pdf)
. Because minor modifications of the orginial code are necessary, a forked version (https://github.com/Wagner-L/Mask_RCNN) is used here.



## Training data

Training and validation image tiles are created based on thes script provided in the folder "pre-processing" and separated in different folders. Image annotations are created for both folders sepreately using the [VGG Image Annotator](ttps://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). The annotation file is in .json format and contains information about bounding boxes and masks of all objects in all images.

![VGG annotator](https://github.com/Wagner-L/ice_wedge_polygons/blob/main/images/VGG_annotator.png)


The required structure of the training/validation dataset has to look as follows:

dataset<br />
 --train<br />
  ----first_image.jpg<br />
  ----second_image.jpg<br />
  ----...<br />
  ----via_project.json<br />
 --val<br />
  ----third_image.jpg<br />
  ----fourth_image.jpg<br />
  ----...<br />
  ----via_project.json<br />


## Training

Processing is done in Google Colab. The dataset for training has to be uploaded in a connected Google drive folder.

The following steps are performed:
1. Importing and cloning repositories
2. Installing the correct versions of libraries (older versions of keras and tensorflow required)
3. Configuration of training
4. Creating dataset class
5. Loading training and validation datasets
6. Training

The trained model has to be saved to serve as input for prediction.

A trained model is accessible here: https://wolke.upgradetofreedom.de/s/Z5wtnjgWbXDT7F2


## Prediction

The following steps are performed:
1. Configuration for predicting
2. Loading the model and weights
3. Load image
4. Predict bounding box and mask
5. Display of results


### Results

Example of prediction for one tile

[result_example](https://github.com/Wagner-L/ice_wedge_polygons/blob/main/images/result_example.png)
