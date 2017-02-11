# read driving_log in as pandas.dataframe
# use given_data set to train.

import pandas as pd
import numpy as np
import math
import cv2
from sklearn.utils import shuffle

# I put the given training data in data_given folder
datapath = 'data_given/'
data = pd.read_csv(datapath + 'driving_log.csv',delimiter=',')

# data augmentation
# I have 3 levels of data augmentation: brightness augmentation, shadow augmentation and locaiton jittering

# 1. Brightness augmentation
def brightness_jitter(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    bright_jitter_rate = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*bright_jitter_rate
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# 2. Shadow augmentation
# random shadows are cast across the image by choosing one random point on the top edge and one random point
# on the bottom edge of the image, connecting the two random selected points and shading all points on
# a random side of the image.
def shadow_jitter(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# 3. Horizontal and vertical shifts
# Shift the camera images horizontally to simulate the effect of car being at different positions on the road.
# Also shift the images vertically by a random number to simulate the effect of driving up or down the slope.
def steering_jitter(image,steer,trans_range):
    # Translation
    rows,cols,cha = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

# cut and resize image
new_size_col,new_size_row=64,64
def preprocessImage(image):
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)
    #image = image/255.-.5
    return image

# assign steering angle to imagges taken from left and right cameras
def preprocess_image_file_train(line_data):
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = line_data['left'][0].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_data['right'][0].strip()
        shift_ang = -.25
    y_steer = line_data['steering'][0] + shift_ang
    image = cv2.imread(datapath+path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,y_steer = steering_jitter(image,y_steer,100)
    image = brightness_jitter(image)
    image = shadow_jitter(image)
    image = preprocessImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    
    return image,y_steer

# Use kera’s generator function to sample images such that images with
# lower angles have lower probability of getting represented in the data set.
def generate_train_from_PD_batch(data,batch_size = 32):
    
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        data = shuffle(data)
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            
            keep_pr = 0
            #x,y = preprocess_image_file_train(line_data)
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
#return batch_images, batch_steering

# construct model
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Input, Dropout, MaxPooling2D, Activation, Cropping2D, Lambda
from keras.models import model_from_json
from keras.activations import relu, softmax
from keras.layers.advanced_activations import ELU

model = Sequential()

#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)
#The example above crops:
#50 rows pixels from the top of the image
#20 rows pixels from the bottom of the image
#0 columns of pixels from the left of the image
#0 columns of pixels from the right of the image
#model.add(Lambda(lambda x: cv2.resize(x,(64,64),interpolation=cv2.INTER_AREA)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
model.add(Convolution2D(24, 5, 5,subsample = (2,2)))
model.add(ELU())
#model.add(Activation('relu'))
# (none, 30, 30, 24)
model.add(Convolution2D(36, 5, 5,subsample = (2,2)))
model.add(ELU())
#model.add(Activation('relu'))
# (none, 13, 13, 36)
model.add(Convolution2D(48, 5, 5,subsample = (2,2)))
model.add(ELU())
#model.add(Activation('relu'))
# (none, 5, 5, 48)
model.add(Convolution2D(64, 3, 3,subsample = (1,1)))
model.add(ELU())
#model.add(Activation('relu'))
# (none, 3, 3, 64)
model.add(Convolution2D(96, 3, 3,subsample = (1,1)))
model.add(ELU())
#model.add(Activation('relu'))
# (none, 1, 1, 96)
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
#model.add(Activation('relu'))
model.add(Dense(50))
model.add(ELU())
#model.add(Activation('relu'))
model.add(Dense(10))
model.add(ELU())
#model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


# train
import numpy as np

val_size = 1
pr_threshold = 1
batch_size = 1024
for i_pr in range(8):
    train_r_generator = generate_train_from_PD_batch(data,batch_size)
    
    nb_vals = np.round(len(data)/val_size)-1
    model.fit_generator(train_r_generator,samples_per_epoch=30000, nb_epoch=1,verbose=1)
    pr_threshold = 1/(i_pr+1)*1

# save model
from keras.models import load_model
model.save('model_test.h5')

from keras.utils.visualize_util import plot
plot(model, to_file='model_test.png')
