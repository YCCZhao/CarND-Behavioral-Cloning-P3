# Behavioral Cloning Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_41_26_528.jpg "Center Image"
[image3]: ./examples/left_2016_12_01_13_41_26_528.jpg "Left Image"
[image4]: ./examples/right_2016_12_01_13_41_26_528.jpg "Right Image"
[image5]: ./examples/center_2017_07_30_09_09_28_033.jpg "Recovery Image1"
[image6]: ./examples/center_2017_07_30_09_09_28_934.jpg "Recovery Image2"
[image7]: ./examples/flip_center_2016_12_01_13_41_26_528.jpg "Flipped Image"
[image8]: ./examples/Training.png "Training Capture"

# Write-up
---

## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](.drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [final_run.mp4](./final_run.mp4)  a video recording of vehicle driving autonomously around the track for one full lap
* And this write-up

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

I chosed [Nvidia CNN network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which consists of 5 layers of convolution neural networks and 4 layers of fully connected neural networks.

|Layers|Properties|
|:-------|:-----------|
|conv| 24 5x5 filters with stride of 2; l2 regularizer of 0.001|
|elu| Activation|
|conv| 36 5x5 filters with stride of 2|
|elu| Activation|
|conv| 48 5x5 filters with stride of 2|
|elu| Activation|
|conv| 64 3x3 filters with stride of 1|
|elu| Activation|
|conv| 64 3x3 filters with stride of 1|
|elu| Activation|
|fc| fully connected layers, output 100 hidden layers|
|elu| Activation|
|fc| fully connected layers, output 50 hidden layers|
|elu| Activation|
|fc| fully connected layers, output 10 hidden layers|
|elu| Activation|
|fc| fully connected layers, output steer angle|

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

```sh
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
```

### 2. Attempts to reduce overfitting in the model

The model contains 50% dropout at fully connected layers in order to reduce overfitting.

```sh
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(act)
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(act)
model.add(Dense(10))
model.add(Dropout(0.5))
```
The model also contains a L2 regularizer with an alpha of 0.001 at the first CNN layer to reduce overfitting. This alpha value was chosen based on validation performance.

```sh
model.add(Convolution2D(24,5,5,subsample=(2,2),W_regularizer=l2(0.001)))
```

The model was trained and validated on different data sets to ensure that the model was not overfitting. 25% data was used for validatoin

```sh
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines) 
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```sh
model.compile(loss='mse', optimizer='adam')
```

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
Since most of the time, the car is driving straight. Measurement of zero degree dominates the training data. Model trained using these data would most likely driving straight, even at the time it shouldn't. To solve this program, 95% of the training data with zero steer angle were removed. Now the training data is normally distributed with zero mean.

```sh
if (float(line[3]) < 0.1 and float(line[3]) > -0.1) and random.uniform(0,1) > 0.05:
  continue
```

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use appropriate training data to train an existing architecture. Then fine tune it.

My first step was to use a convolutional neural network model similar to the [NVIDIA CNN network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because it was used in the real world environment and showed some promising results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added 50% dropout at the fully connected layers. With dropout the mean squared error on the validation was decreased. But I wanted to continue improving the model, so I added 0.001 L2 regularizers at convolutional layers. Both training error, and validation error were increased. When I ran the simulator, the car was not very stable. It hit the wall while driving on the bridge. Then I modified the model by only having L2 regularizer at the first convolutional layers.

This time when I run the simulator, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

I chosed [NVIDIA CNN network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which consists of 5 layers of convolution neural networks and 4 layers of fully connected neural networks.

```sh
model = Sequential()
act = ELU(alpha=0.05)
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),W_regularizer=l2(0.001)))
model.add(act)
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(act)
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(act)
model.add(Convolution2D(64,3,3))
model.add(act)
model.add(Convolution2D(64,3,3))
model.add(act)
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(act)
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(act)
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(act)
model.add(Dense(1)
```

Here is a visualization of the architecture.

![alt text][image1]

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I also included data provided in class. Here is an example image of center lane driving:

![alt text][image2]

Also use left and right images

![alt text][image3]
![alt text][image4]

For left and right images, a correction factor was used to calculated the correct angle if images were actual center images. A random number from -0.05 to 0.05 is added to the correction factor, so that measurement remained normally distributed.

```sh
correction = 0.2 + random.uniform(-0.05,0.05)
measurements.append(measurement+correction)
measurements.append(measurement-correction)
```

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to return on track when the route is off. 

To augment the data sat, I also flipped images and angles thinking that this would ensure model have similar amount of training data of turning each direction.
For example, here is an image that has then been flipped:

![alt text][image5]

Sign of the measurements are flipped as well.

```sh
measurements.append(-measurement)
```

After the collection process, I had 12752 number of data points. I then preprocessed this data by normalization. Now training data have value from -0.5 to 0.5. I also cropped the images to exclude noise such as sky, background, and car hook.
I finally randomly shuffled the data set and put 25% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the decreased rate of mean squared error improvement. I used an adam optimizer so that manually training the learning rate wasn't necessary.
