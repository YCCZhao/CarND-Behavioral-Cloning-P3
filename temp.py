# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:43:00 2017

@author: Yunshi_Zhao
"""

import csv
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

lines = []
#files = ['../sim_data/driving_log.csv',
#         '../sim_data/driving_log_recovery.csv',
#         '../sim_data/driving_log_recovery2.csv',
#         '../sim_data/driving_log_sample.csv']

files = ['../sim_data/driving_log_sample.csv']

for file in files:
    with open(file) as f:
        reader = csv.reader(f)
        for line in reader:
            if (float(line[3]) < 0.1 and float(line[3]) > -0.1) and random.uniform(0,1) > 0.05:
                continue
            lines.append(line)
images = []
measurements = []  

for line in lines:
  
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

fit = plt.hist(y_train,bins=24)
fig.savegit('./dist.png')