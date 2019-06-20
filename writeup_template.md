# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the Nvidia CNN (model.py lines 125-137). The network consists of 5 convolutional layers, where 3 layers have a 5x5 filter and 2 layers have a 3x3 filter. All 5 convolutional layers have RELU activation layers following them to introduce nonlinearity. The data is then flattened and moved through a 50% dropout layer to regularize and prevent overfitting. Then 4 dense layers with sizes 1164, 100, 50, and finally 1 to output the steering adjustment.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 133). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 48). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 139).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I included three laps around the track, and one lap in the opposite direction. In spots where there were tight turns, I included training samples that would have the car recover toward the center if it started off too much to the side.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build off of the previous project. Initially, I used the LeNet architecture, but the car drove very jittery and went off the road frequently.

Then I tried using the Nvidia architecture since the research paper had shown that it was used for a very similar task. I ran into memory errors when attempting to use all the data for training at once. So, I decided to use a generator to make batch updates to the model. For many train/validation/tests, I had improperly generated 6 images per single line (center, left, right cameras and the horizontal flips of each). This caused the car to veer off the road on some turns, especially where the road edge pattern would change. So, instead the generator randomly chooses a camera, then randomly decides whether to flip the image and measurement. This finally led to a successful run.

The model was trained for 4 epochs because both the training loss and validation loss did not appear to improve on more epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The architecture has been discussed above, except the preprocessing. In order to speed up training, I chose to preprocess the images by cropping, normalizing, and resizing the images to fit the expected Nvidia CNN (model.py lines 89-96). The preprocess function was called in the generator when the camera to use for that sample was chosen.

The training set considered for batch creation was filtered to include speeds where the car is going at least 5 mph (model.py line 13). This was done because while collecting training samples in the simulator, I would need to press the record button then align my cursor to drive. I wanted to exclude any images where I was still getting ready to drive, and by 5 mph I would expect myself to be in a comfortable trajectory.

The above two modifications helped create a model that could drive the first course. Unfortunately the second course requires more modifications. There may be either more training sets needed, or more preprocessing to handle the shadows, limited horizon on hills, and center line.
