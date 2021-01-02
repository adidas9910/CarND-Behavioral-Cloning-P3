# **Behavioral Cloning Project** 

## ChangYuan Liu

### This document is a brief summary and reflection on the project.

The goals / steps of this project are the following:

* Collect training dataset

* Design a Model Architecture and Train the Model
  * Pre-process the data, including augmentation and normalization.
  * Utilize the deep learning model from Nvidia team.
  * Train the model.

* Test the Model in the Autonomous Mode in the Simulator
  * Run the simulator with the trained model
  * Produce the video

* Summarize the Project
  
[//]: # (Image References)
[image1]: ./writeup_images/1_image_example.png
[image2]: ./writeup_images/2_measurements.png
[image3]: ./writeup_images/3_cnn-architecture-624x890.png
[image4]: ./writeup_images/4_train_history.png

---


## 1. Collect training dataset

After the simulation environment is set up, the first step of the project is collecting the dataset for training. After a few failing tries, I realized the deep leaning model could do as goog as the training set itself. So I extended the training data to a few laps, and I behaved with some waggling and corrections. 

For this project, I collected 61,764 images, including the images from left, center, and right cameras.

## 2. Design a Model Architecture and Train the Model

### 2.1. Pre-process the data
First, augment the dataset to generalize situations. Two augmentations are used in this project: (1) flipping the images and measurements; (2) using the images from left and right cameras and applying corrections to those images. After the augmentaion, there are 123,528 samples of images with shape of (160, 320, 3).

Then, it is important to explore the dataset by plotting example images and the measurements. Plotting measurements helped me to find out that it was a bad idea to use keyborad to control directions in the training mode as the keyboard only provides a few anlges in the range between -1 and 1. Using mouse to control the direction would provide much finer angle data than keyboard.

![alt text][image1]

![alt text][image2]

### 2.2. Utilize the deep learning model from Nvidia team

[The deep learning model from Nvidia](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) is used in this project. Its architecture is showing as below:

![alt text][image3]

The following changes are made to the model for this project:

**(1) A cropping layer is added between the normalization layer and the first convolutional layer;**

**(2) The output put layer is changed to one node.**

### 2.3. Train the model
Train the model on the dataset which is splitted into training, and validation data. The main knobs are the following hyperparameters: 

    lr = 2.0e-3 #learning rate used for Adam optimizer
    epochs = 4 # number of epochs

Here is the history of Mean Squared Error (mse) over the epochs:

![alt text][image4]

After many iterations, I found out that the number of epochs could be smaller. Even it was set to more than 10, it might help reduce mse of the training set but not the validation set. In this case, setting the number of epochs to smaller would actually prevent overfitting.

## 3. Test the Model in the Autonomous Mode in the Simulator
### Run the simulator with the trained model
Run the cmd 'python drive.py model runx' and enter the autonomous mode in the simulator. The images needed to produce video would be generated in the folder 'runx'.
### Produce the video
Run the cmd 'python video.py runx'. The video 'runx.mp4' would be produced from the images in the folder 'runx'.

## 4. Summarize the Project
### 4.1. Shortcoming

A shortcoming is the fact that lots of efforts and time are needed to generate the training dataset. Only after a few days, I realized that the training dataset is so important. There should be enough data points in the training dataset, and enough valuable or 'noisy' data in the dataset.

Another shortcoming is that the project is about only the angle. The driving task is much more than that, at least there should some aspects about the speed, etc. 

### 4.2. Possible improvements

A possible improvement would be tweaking the model by adding or modifying some layers. It may include adding dropout, pooling etc.

Other possible improvements may include getting more training dataset and finding better ways to tune the hyperparameters.

Aside from the project itself, some improvements are needed for the code and simulation for the project from Udacity. The code and simulation environment are not up-to-date with the newer version of Python and tensorflow, etc. The workspace is not so user-friendly. I spent more than half of the time to figure out how to run the code and simulation on my laptop.


