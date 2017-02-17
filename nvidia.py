import os
import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#height = 80
#width = 160
height = 160
width = 320

dataSubDir = 'data'
dataLog = 'driving_log.csv'
dataDir = os.path.join('data', dataSubDir)

samples = []
with open(os.path.join(dataDir,dataLog)) as csvfile:
    samples = list(csv.reader(csvfile))


#train_samples = samples
#validation_samples = samples
train_samples, validation_samples = train_test_split(samples[1:])
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.join(dataDir,batch_sample[0].strip())
                center_img = cv2.imread(name)
                center_image = center_img
                #center_image = cv2.resize(center_img,(width,height))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #yield sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
print('generators setup')

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(height, width, 3)))
model.add(Lambda(lambda x: x/127.5 - 1))
model.add(Conv2D(24,5,5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36,5,5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48,5,5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,3,3))
model.add(Conv2D(64,3,3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.001), loss='mse' , metrics=['accuracy'])
#model.summary()

history = model.fit_generator(train_generator, \
                              samples_per_epoch = len(train_samples), \
                              validation_data=validation_generator, \
                              nb_val_samples=len(validation_samples), nb_epoch=10) #, verbose=0)
