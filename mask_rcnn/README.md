
# Mask RCNN implementation to detect ice wedge polygons


Mask R-CNN is intended to perform instance segmentation,that is detecting bounding boxes and segmentation masks for each instance of an object in an image.

The code is based on https://github.com/matterport/Mask_RCNN using Python 3, Keras and TensorFlow using  a Feature Pyramid Network (FPN) and a ResNet101 backbone (based on this paper). Because minor modifications of the orginial code are necessary, a forked version (https://github.com/Wagner-L/Mask_RCNN) is used here.


## Training data

Training and validation image tiles are created based on thes script provided in the folder "pre-processing" and separated in different folders. Image annotations are created for both folders sepreately using the [VGG Image Annotator](ttps://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). The annotation file is in .json format and contains information about bounding boxes and masks of all objects in all images.

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


## Processing

Processing is done in Google Colab.

## Results

![name-of-you-image](https://your-copied-image-address)
