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
[image4]: ./examples/right_2016_12_01_13_41_26_528.jng "Right Image"
[image5]: ./examples/center_2017_07_30_09_09_28_033.jng "Recovery Image"
[image6]: ./examples/flip_center_2016_12_01_13_41_26_528.jpg "Flipped Image"

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

The model contains dropout layers in order to reduce overfitting.

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

```sh
model.add(Convolution2D(24,5,5,subsample=(2,2),W_regularizer=l2(0.001)))
```

The model was trained and validated on different data sets to ensure that the model was not overfitting.

```sh
train_samples, validation_samples = train_test_split(lines) 
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```sh
model.compile(loss='mse', optimizer='adam')
```

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

```sh
if (float(line[3]) < 0.1 and float(line[3]) > -0.1) and random.uniform(0,1) > 0.05:
  continue
```

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

I chosed [Nvidia CNN network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which consists of 5 layers of convolution neural networks and 4 layers of fully connected neural networks.

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

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image2]

Also use left and right images
![alt text][image3]
![alt text][image4]
Correction

```sh
correction = 0.2 + random.uniform(-0.05,0.05)
measurements.append(measurement+correction)
measurements.append(measurement-correction)
```

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
![alt text][image5]

```sh
measurements.append(-measurement)
```

After the collection process, I had 11659 number of data points. I then preprocessed this data by ...
I finally randomly shuffled the data set and put Y% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
