
import numpy as np
import pandas as pd


from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical

from keras import layers
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D , Flatten, Conv2D, ZeroPadding2D
from keras.models import Sequential

from keras.optimizers import RMSprop , SGD, Adam
from keras.callbacks import ModelCheckpoint

import keras.backend as K

K.set_image_data_format('channels_last')



def research_paper_model_mod(input_shape, classes = None, dropout_rate = 0.2):
    model = Sequential()
    
     
    model.add(Conv2D(input_shape = input_shape,  strides=(4,4) ,filters = 96 , kernel_size = (7,7), name= "conv_1"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2) , name = "max_pool_1"))
    model.add(BatchNormalization(axis = 3 , name ="bn_1") )
    
    model.add(ZeroPadding2D(padding=(2, 2), name= "zpad_1"))
    model.add(Conv2D(strides=(1,1) ,filters = 128 , kernel_size = (5,5), name= "conv_2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2) , name = "max_pool_2"))
    model.add(BatchNormalization(axis = 3 , name ="bn_2") )

    model.add(ZeroPadding2D(padding=(1, 1), name= "zpad_2"))
    model.add(Conv2D(strides=(1,1) ,filters = 128 , kernel_size = (3,3), name= "conv_3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2) , name = "max_pool_3"))
    model.add(BatchNormalization(axis = 3 , name ="bn_3") )
    
    model.add(ZeroPadding2D(padding=(1, 1), name= "zpad_3"))
    model.add(Conv2D(strides=(2,2) ,filters = 256 , kernel_size = (3,3), name= "conv_4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2) , name = "max_pool_4"))
    model.add(BatchNormalization(axis = 3 , name ="bn_4") )

    model.add(Flatten())

    model.add(Dense(128 , name = 'Fully_connected_4' ))
    model.add(Activation('relu'))
    model.add(Dropout(rate = dropout_rate) )
    
    #model.add(Dense(64 , name = 'Fully_connected_5' ))
    #model.add(Activation('relu'))
    #model.add(Dropout(rate = dropout_rate) )

    model.add(Dense(classes, activation= 'softmax', name = 'softmax_6' ))

    return model









