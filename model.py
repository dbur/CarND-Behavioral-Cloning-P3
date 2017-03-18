import csv
import cv2
import numpy as np
# import progressbar


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # exclude where going slow or no steering adjustment happens
        # these are probably areas when i start recording
        if float(line[6]) > 5.0: # and abs(float(line[3])) > 0.01:
            lines.append(line)

images = []
measurements  = []
# for left and right camera angles
correction = 0.25
# bar = progressbar.ProgressBar()

#########################
# for full dataset
# for line in bar(lines):
#     for i in range(3):
#         image = cv2.imread(line[i])
#         images.append(image)
#         measurement = float(line[3])
#         if i == 1:
#             # left camera
#             measurement += correction
#         if i == 2:
#             # right camera
#             measurement += -1*correction
#         measurements.append(measurement)
#         images.append(cv2.flip(image,1))
#         measurements.append(measurement*-1.0)

# X_train = np.array(images)
# y_train = np.array(measurements)
#
#########################

#########################
# for samples dataset
from sklearn.model_selection import train_test_split
import sklearn
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
def generator(samples, batch_size=32):
    num_samples = len(lines)
    while 1:
        sklearn.utils.shuffle(lines)
        for offset in range(0,num_samples,batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                i = np.random.randint(3)
                image = cv2.imread(batch_sample[i], cv2.COLOR_BGR2RGB)
                # preprocess images outside of model
                image = np_preprocess_image(image)
                measurement = float(batch_sample[3])
                if i == 1:
                    # left camera
                    measurement += correction
                if i == 2:
                    # right camera
                    measurement += -1*correction
                # randomly flip the image
                if np.random.randint(2) > 0:
                    images.append(image)
                    measurements.append(measurement)
                else:
                    images.append(cv2.flip(image,1))
                    measurements.append(measurement*-1.0)
                X_train = np.array(images)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train,y_train)

#
#########################

def model_preprocess_image(x):
    import tensorflow as tf
    x = x / 255.0 - 0.5
    return tf.image.resize_images(x,(66,200))

def np_preprocess_image(img):
    # crop to 80,320
    img = img[60:-20,:,:]
    # normalize to [-0.5,0.5]
    img = img / 255.0 - 0.5
    # resize to 80,200
    img = cv2.resize(img,(200,66))
    return img



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
# model preprocessing:
# model.add(Cropping2D(cropping=((60,25),(0,0)), input_shape=(160,320,3)))
# model.add(Lambda(model_preprocess_image))



# # lenet:
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# nvidia architecture:
# preprocessed images:
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu',input_shape=(66,200,3)))
# model proprocessed layer:
# model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# fit for full dataset
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

# fit_generator for batch sampling
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
    validation_data=validation_generator, nb_val_samples=len(validation_samples),nb_epoch=4)

model.save('model.h5')

# Flow device (/gpu:0) -> (device: 0, name: Quadro M2000M, pci bus id: 0000:01:00.0)
# 16947/16947 [==============================] - 1236s - loss: 0.0107 - val_loss: 0.0164
# Epoch 2/5
# 16947/16947 [==============================] - 1170s - loss: 0.0027 - val_loss: 0.0359
# Epoch 3/5
# 16947/16947 [==============================] - 1051s - loss: 0.0030 - val_loss: 0.0181
# Epoch 4/5
# 16947/16947 [==============================] - 1045s - loss: 0.0029 - val_loss: 0.0276
# Epoch 5/5
# 16947/16947 [==============================] - 1193s - loss: 0.0035 - val_loss: 0.0237

from keras import backend as K
K.clear_session()
