# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. You trained, validated and tested a model using Keras. The model ouputs a steering angle to an autonomous vehicle.

I used the simulator provided to steer a car around a track for data collection. Then I used image data and steering angles to train a neural network and usedthis model to drive the car autonomously around the track.

The project submission includes: 
* [model.py](./model.py) (script used to create and train the model)
* [drive.py](./drive.py) (script to drive the car - feel free to modify this file)
* [model.h5](./model.h5) (a trained Keras model)
* a report writeup
* [final_run.mp4](./final_run.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

Dependencies
---
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

