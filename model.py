# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:38:19 2017

@author: Yunshi_Zhao
"""
import csv
import cv2
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split

lines = []
files = ['../sim_data/driving_log.csv',
         '../sim_data/driving_log_recovery.csv',
         '../sim_data/driving_log_recovery2.csv',
         '../sim_data/driving_log_sample.csv']

for file in files:
    with open(file) as f:
        reader = csv.reader(f)
        for line in reader:
            if (float(line[3]) < 0.1 and float(line[3]) > -0.1) and random.uniform(0,1) > 0.05:
                continue
            lines.append(line)

        
train_samples, validation_samples = train_test_split(lines) 

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_sample = samples[offset:offset+batch_size]
            
            images = []
            measurements = []    
            for line in batch_sample:
                measurement = float(line[3])
                correction = 0.2 + random.uniform(-0.05,0.05)
                center = line[0].split("\\")[-1]
                left   = line[1].split("\\")[-1]
                right  = line[2].split("\\")[-1]
                img_center = cv2.imread('../sim_data/IMG/'+center)
                img_left   = cv2.imread('../sim_data/IMG/'+left)
                img_right  = cv2.imread('../sim_data/IMG/'+right)
                img_flip = np.fliplr(img_center)
                  
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                images.append(img_flip)
                
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)
                measurements.append(-measurement)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

model = Sequential()
#act = LeakyReLU(alpha=0.05)
#act = Activation('relu')
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
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    samples_per_epoch=len(train_samples)*4,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*4, 
                    nb_epoch=5)

model.save('model.h5')
exit()