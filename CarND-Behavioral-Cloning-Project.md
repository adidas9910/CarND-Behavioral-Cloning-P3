# **Traffic Sign Classifier Project** 

## ChangYuan Liu

### This document is a brief summary and reflection on the project.

The goals / steps of this project are the following:

* Dataset Summary & Exploration
  * Explore the datasets for training, validation, and testing
  * Virtualize the distribution of the datasets
* Design and Test a Model Architecture
  * Pre-process the images, including grayscaling and normalization
  * Revise LeNet deep learning model
  * Train the model
* Test the Model on New Images
  * Predict the sign type for each image
  * Analyze performance
  * Output top 5 softmax probabilities for each image 

[//]: # (Image References)
[image1]: ./writeup_images/1_data_distribution.png
[image2]: ./writeup_images/2_image_example.png
[image3]: ./writeup_images/3_lenet.png
[image4]: ./writeup_images/4_history_of_accuracy.png
[image5]: ./writeup_images/5_new_images.png
[image6]: ./writeup_images/6_normalized_images.png

---


## 1. Dataset Summary & Exploration

First, load the datasets for training, validation, and testing from pickle files.

    Number of training examples = 34799
    Number of validation examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43

Then, virtulize the distribution of the datasets. The datasets seem having similar distribution in different categorical labels.

Here is the distribution of those 43 labels in these three datasets:

![alt text][image1]



## 2. Design and Test a Model Architecture

(1) Pre-process the images in the datasets, including converting the images into grayscale and normalizing them. Here is an example of the original image and normalized image.

![alt text][image2]

(2) Revise the LeNet model. 

Because the problem in this project is similar to the number identification problem, the LeNet is used for the project. Its diagram is shown as below. The model used in this project has different number of nodes in the last three layers: the number of nodes in layer 3 is changed from 120 to 240; the number of nodes in layer 4 is changed from 84 to 168; the number of layer 5 is changed from 10 to 43. Because the number of outputs is 43, which is much bigger than 10 in the number identification problem, I think it would help to expand the last a few layers.

![alt text][image3]

**Input**

The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. If the images are grayscale, C is 1.

**Architecture**

**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** relu activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** relu activation function.

**Pooling.** The output shape should be 5x5x16.

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

**Layer 3: Fully Connected.** This should have 120 outputs. I changed it to 240 outputs for this project.

**Activation.** relu activation function.

**Layer 4: Fully Connected.** This should have 84 outputs. I changed it to 168 outputs for this project.

**Activation.** relu activation function.

**Layer 5: Fully Connected (Logits).** This should have 10 outputs. I changed it to 43 outputs for this project.


**Output**

Return the result of the 2nd fully connected layer (Layer 5).

(3) Train the model.
Train the model on the dataset for training, and validate on the dataset for validation. The main knobs are the following hyperparameters: 

    LEARNING_RATE = 7.5e-4 #learning rate used for Adam optimizer
    LAMBDA = 7.0e-5 # regularization parameter
    EPOCHS = 60 # number of epochs
    BATCH_SIZE = 128 # batch size

Here is the history of accuracy over the epochs:

![alt text][image4]

## 3. Test a Model on New Images
(1) Pre-process the new images.

The original new images have different sizes, so they are resized and normalized as below before they are tested.

![alt text][image5]

![alt text][image6]

The correct label for those images should be:
                         
    ClassId              SignName                  
    27                Pedestrians
    18            General caution
    1        Speed limit (30km/h)
    25                  Road work
    28          Children crossing
                     
But the predictions are:

    ClassId                                           SignName
    12                                           Priority road
    14                                                    Stop
    42       End of no passing by vehicles over 3.5 metric ...
    25                                               Road work
    12                                           Priority road
The accuracy is only 20%. In addition, the top 5 softmax probabilities indicate that all the wrong predictions are very certain as their highest probilities are close to 1.

    TopKV2(values=array(
      [[9.9999714e-01, 1.5180156e-06, 7.2078274e-07, 4.5569141e-07, 7.5943134e-08],
       [9.9934989e-01, 5.7708321e-04, 6.8868154e-05, 4.1291346e-06, 8.1953768e-09],
       [9.9853671e-01, 9.8147546e-04, 4.7962563e-04, 6.2025799e-07, 5.6595144e-07],
       [8.9593124e-01, 5.6267273e-02, 4.7791585e-02, 4.5822853e-06, 2.8378727e-06],
       [9.9830127e-01, 1.6987451e-03, 3.1564407e-09, 4.2253191e-11, 8.7820655e-15]], 
       dtype=float32), indices=array(
      [[12, 25, 13, 15, 38],
       [14, 17, 12, 40, 38],
       [42, 12,  1, 38, 25],
       [25, 11, 30, 34, 31],
       [12, 40, 42, 41, 17]]))


Overall, the model couldn't predict the new images well.

## 4. Shortcoming and Possible improvements

The first shortcoming is the fact that the performance of the model for identifying new images from web is poor. This indicates that it is not practical to use this model in real systems.

Another shortcoming is that the training process is tedious and time-consuming for tuning the hyperparameters. The number of epochs is high in my case; the initial conditions might need to be improved.

A possible improvement would be tweaking the model by adding or modifying some layers. It may include adding dropout, changing activation functions (such as sigmoid), etc.

Other possible improvements may include getting the training dataset more even distributed, transforming or rotating images, and finding better ways to tune the hyperparameters.


