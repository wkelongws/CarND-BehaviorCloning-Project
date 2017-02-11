#**Behavioral Cloning** 

##Udacity Self-driving-Car Nano-degree Term1 Porject 3 

###Train a car to drive autonomously in a driving simulator

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./bright_jitter.png "brightjitter"
[image2]: ./locationjitter.png "locationjitter"
[image3]: ./shadowjitter.png "shadowjitter"
[image4]: ./left_center_right.png "leftright"
[image5]: ./trainingsample.png "trainingsample"
[image6]: ./model.png "model"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows how I preprocess the data and describes the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I have tried several different structures. Some randomly chosed structure I tried predicts nearly constant steering angle. I think it may be because of the structure is not deep enough. It turns out that the Nvidia structure works for me. The difference is I used 64*64*3 input size rather than 66*200*3 used in the Nvidia paper. So as conciqunence, the number of parameters in each leayer are different. 

So my model has 9 layers including the output layer, 5 convolutional layers and 4 fully connected layers. Each leayer and its following activation are showed below:

Input layer: 64*64*3 image from the center camera
Layer 1: 2Dconvolutional, kernal = (5,5), stride = (2,2), 24 filters
ReLU activation
feature size = 30*30*24
Layer 2: 2Dconvolutional, kernal = (5,5), stride = (2,2), 36 filters, output size = 13*13*36
ReLU activation
feature size = 13*13*36
Layer 3: 2Dconvolutional, kernal = (5,5), stride = (2,2), 48 filters, output size = 5*5*48
ReLU activation
feature size = 5*5*48
Layer 4: 2Dconvolutional, kernal = (3,3), stride = (1,1), 64 filters, output size = 3*3*64
ReLU activation
feature size = 3*3*64
Layer 5: 2Dconvolutional, kernal = (3,3), stride = (1,1), 96 filters, output size = 1*1*96
ReLU activation
feature size = 1*1*96
Layer 6: Fully connected layer, output = 100
ReLU activation
feature size = 100,
Layer 6: Fully connected layer, output = 50
ReLU activation
feature size = 50,
Layer 6: Fully connected layer, output = 10
ReLU activation
feature size = 10,
Layer 6: Fully connected layer/output layer, output = 1
Linear activation
output = steering angle

####2. Attempts to reduce overfitting in the model

I have tried to add dropouts and BatchNormalization into each hidden layer to prevent overfitting.
I have tried to add dropouts only, BatchNormalization only and both dropouts and BatchNormalization. But unfortunately none of those trials looks good to me.. After indroducing dropouts and BatchNormalization, it takes more time to train the model and the trained models cann't drive the car well in the simulator. 

So eventally I decided to not use dropouts or BatchNormalization, instead, to prevent overfitting I used early termination in my model training. So I only trained the model for 1 epoch with verbose = 1. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The result is: the model can drive the car in lane on track 1 for hours and drive the car quite smoothly on track 2 also.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
As described in the above block, I trained the model for only 1 epoch to prevent overfitting.
I used keras fit_generator to generate training batches while training instead of storing all training samples in memory.
for training samples per epoch I have tried different values and there is not much difference whether it is 20000 or 30000 as long as it is reasonably large. I chose 30000 samples per epoch.
Since my machine has enough memory, I chose batch size to be 1024 to sort of increase the converging rate.


####4. Appropriate training data

First of all, I only used the training data set given by Udacity and there are a couple of reasons why I did that:

  1). It was super hard for me to collect quality data using the provided simulator.

The stable simulator used keyboard to control the steering angle. In this way, I have to quickly click and release to steer and the values of steering angle are just kind of impluses -- in one frame the steering angle is 0 and the next similar frame there could be a large steering angle. I wouldn't call this quality training data. In addtion, I event couldn't keep the car in lane using this control.

The beta simulator enables mouse input. But I don't know why they make the steering angle based on the strenth of mouse moving rather than the mouse distance (by the time I downloaded the simulator). This is a super bad idea and I totally cannot control it.

I found some people in Slack recommended using joysticks for data collection in this project. Sorry I don't have one.

  2). The provided training set look OK.

Someone reconstruct the data collection video based the images provided by Udacity. Based on the video it looks the car was driving OK, at least better than me. Also people on slack confirmed that it is totally possible to finish this project by just using the provided data set, so I decided to use the given data set to train my model.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 


Rather than directly use the given training set, I did a lot of prepocessing to it. Besides images of the center camera, I utilized images from both left and right camera with an additional steering angle attached. I jittered the images in several ways to simulate different driving conditions. The detailed method and be found in train.py.
Some images are shown here to demonstrate the preprocessing:

  1). brightness jitter: randomly change the brightness of the image
  ![alt text][image1]
  
  2). location jitter: Shift the camera images horizontally to simulate the effect of car being at different positions on the road. Also shift the images vertically by a random number to simulate the effect of driving up or down the slope. When shifting horizontally, assign 0.004 steering angle per pixel shift. This is a empirically selected value inspired also by people in Slack. When shifting verically, no additional steering angle will be assigned.
  ![alt text][image2]
  
  3). shadow jitter: random shadows are cast across the image by choosing one random point on the top edge and one random point on the bottom edge of the image, connecting the two random selected points and shading all points on a random side of the image.

  ![alt text][image3]
  
  4). assigning additional steering angles to images from left and right camera: a 0.25 extra steering angle was assigned to images from left and right camera:
  ![alt text][image4]

And combining all these preprocessings, here are some excamples of the preprocessed images which are actually sent to the model to learn:

![alt text][image5]

Eventually, the model can drive the car in lane on track 1 for hours and drive the car quite smoothly on track 2 as well.

here is a visualization of the model structure:
![alt text][image6]
