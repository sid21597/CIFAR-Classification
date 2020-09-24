#!/usr/bin/env python
# coding: utf-8

# Convolution only network

"""
"Obtained average test accuracy - 90.5%"

Architecture : 


Layer (type)                 Output Shape              Param #   
=================================================================
block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792      
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 64)        256       
_________________________________________________________________
activation (Activation)      (None, 32, 32, 64)        0         
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928     
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 64)        0         
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 128)       0         
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
activation_3 (Activation)    (None, 16, 16, 128)       0         
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168    
_________________________________________________________________
batch_normalization_4 (Batch (None, 8, 8, 256)         1024      
_________________________________________________________________
activation_4 (Activation)    (None, 8, 8, 256)         0         
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080    
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 8, 256)         1024      
_________________________________________________________________
activation_5 (Activation)    (None, 8, 8, 256)         0         
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080    
_________________________________________________________________
batch_normalization_6 (Batch (None, 8, 8, 256)         1024      
_________________________________________________________________
activation_6 (Activation)    (None, 8, 8, 256)         0         
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 8, 8, 256)         590080    
_________________________________________________________________
batch_normalization_7 (Batch (None, 8, 8, 256)         1024      
_________________________________________________________________
activation_7 (Activation)    (None, 8, 8, 256)         0         
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 4, 4, 512)         1180160   
_________________________________________________________________
batch_normalization_8 (Batch (None, 4, 4, 512)         2048      
_________________________________________________________________
activation_8 (Activation)    (None, 4, 4, 512)         0         
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
batch_normalization_9 (Batch (None, 4, 4, 512)         2048      
_________________________________________________________________
activation_9 (Activation)    (None, 4, 4, 512)         0         
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
batch_normalization_10 (Batc (None, 4, 4, 512)         2048      
_________________________________________________________________
activation_10 (Activation)   (None, 4, 4, 512)         0         
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
batch_normalization_11 (Batc (None, 4, 4, 512)         2048      
_________________________________________________________________
activation_11 (Activation)   (None, 4, 4, 512)         0         
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 2, 2, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
batch_normalization_12 (Batc (None, 2, 2, 512)         2048      
_________________________________________________________________
activation_12 (Activation)   (None, 2, 2, 512)         0         
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
batch_normalization_13 (Batc (None, 2, 2, 512)         2048      
_________________________________________________________________
activation_13 (Activation)   (None, 2, 2, 512)         0         
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
batch_normalization_14 (Batc (None, 2, 2, 512)         2048      
_________________________________________________________________
activation_14 (Activation)   (None, 2, 2, 512)         0         
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
batch_normalization_15 (Batc (None, 2, 2, 512)         2048      
_________________________________________________________________
activation_15 (Activation)   (None, 2, 2, 512)         0         
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 1, 1, 4096)        2101248   
_________________________________________________________________
batch_normalization_16 (Batc (None, 1, 1, 4096)        16384     
_________________________________________________________________
activation_16 (Activation)   (None, 1, 1, 4096)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 1, 1, 4096)        16781312  
_________________________________________________________________
batch_normalization_17 (Batc (None, 1, 1, 4096)        16384     
_________________________________________________________________
activation_17 (Activation)   (None, 1, 1, 4096)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 1, 1, 10)          40970     
_________________________________________________________________
batch_normalization_18 (Batc (None, 1, 1, 10)          40        
_________________________________________________________________
global_average_pooling2d (Gl (None, 10)                0         
_________________________________________________________________
activation_18 (Activation)   (None, 10)                0         
=================================================================
Total params: 39,002,738
Trainable params: 38,975,326
Non-trainable params: 27,412
_________________________________________________________________

"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 5 of 6 animal classes for training are present (no frog)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler



import os
import math

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.python.util.tf_export import tf_export

batch_size = 32
num_classes = 10
"""
Increased the number of epochs to 200 because, with more epochs, the network will train well and weights can be updated more
Which in turn increase the accuracy

"""
epochs = 200

"""
Switched data augmentation to true.
Data augmentation helps create new data from existing data. More the data, more the accuracy achieved from the network.
As there is more data on which model can train on, accuracy increases.Due to more data, the model can generalize well.

"""
#data_augmentation = True
data_augmentation = True
#num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


"""
Defined a learning rate scheduler which cuts learning rate by half for every 30 epochs starting from 15th epoch until 135th epoch
Restricted it until 135 because, the learning rate became too low and may not reach the global minima
Due to decrease in learning rate, the model can generalize well and it helps model to converge at an optimal solution
With a large learning rate, the model can converge at sub optimal solution and may over-shoot from required convergence point
So, reducing learning rate helps better.

"""
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 15:
        lrate = 0.0005
    elif epoch > 45:
        lrate = 0.0005
    elif epoch > 75:
        lrate = 0.00025
    elif epoch > 105:
        lrate = 0.00025

    return lrate

def load_data5():
  """Loads CIFAR10 dataset. However, just 5 classes, all animals except frog
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
#  dirname = 'cifar-10-batches-py'
#  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#  path = get_file(dirname, origin=origin,  untar=True)
#  path= './cifar-10-batches-py'
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Below shows a test class has 999 examples instead of the claimed 1000
#  tclasscount=np.zeros((10,), dtype=int)
#  for i in range(0, len(y_test)-1):
#    tclasscount[y_test[i][0]]= tclasscount[y_test[i][0]] + 1
#  print('Test class count',tclasscount)
  num_train_samples = 50000
  num_5_class = 25000
  num_5_test = 4999 # should be 5000 if all the categories had 1000 in them but they do not. One is missing.
  print('x_train shape orig:', x_train.shape)
  print('More:', x_train.shape[1:])
  print('y_test shape',y_test.shape)

  x5_train = np.empty((num_5_class, 32, 32, 3), dtype='uint8')
  y5_train = np.empty((num_5_class,), dtype='uint8')

  count=0

  for i in range(0, len(y_train)-1):
   if (y_train[i][0] == 2) or (y_train[i][0] == 3) or (y_train[i][0] == 4) or (y_train[i][0] == 5) or (y_train[i][0] == 7):
    x5_train[count]=x_train[i]
    y5_train[count]=y_train[i]
    count=count+1
   
    # find test data of interest
  count=0
  x5_test=np.empty((num_5_test, 32, 32, 3), dtype='uint8')
  y5_test= np.empty((num_5_test,), dtype='uint8')

  for i in range(0, len(y_test)-1):
   if (y_test[i][0] == 2) or (y_test[i][0] == 3) or (y_test[i][0] == 4) or (y_test[i][0] == 5) or (y_test[i][0] == 7):
    x5_test[count]=x_test[i]
    y5_test[count]=y_test[i]
    count=count+1
# Below shows class 7 is only 999 and not 1000 examples!!!  One horse got away it seems.
#    if(y_test[i][0] == 2):
#     c2=c2+1
#    if(y_test[i][0] == 3):
#     c3=c3+1
#    if(y_test[i][0] == 4):
#     c4=c4+1
#    if(y_test[i][0] == 5):
#     c5=c5+1
#    if(y_test[i][0] == 7):
#     c7=c7+1
#  print('c2count, c3count, c4count, c5count, c7count',c2,c3,c3,c5,c7)
#  print('y5tstshape',y5_test.shape, count)
#  print('y5tst',y5_test)
#  return (x_train, y_train), (x_test, y_test)
  return (x5_train, y5_train), (x5_test, y5_test)



(x_train, y_train), (x_test, y_test) = load_data5()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

steps_for_epoch = math.ceil(x_train.shape[0] / batch_size)
print('num classes',num_classes)
print('y_train',y_train)
print('y_test',y_test)
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

"""
Added a completely custom and new convolution only network which is VGG19. I have edited VGG19 network.
I have replaced dense layers with their corresponding convolution only layers.

The VGG19 network has been referenced from :  https://github.com/BIGBALLON/cifar-10-cnn/blob/master/3_Vgg19_Network/Vgg19_keras.py

I have modified the layer by removing kernel regularizer and changing the activation function from relu to elu
ELU(Exponential linear unit) activation function is used instead of Relu as Elu curves smoothly compared to sharp curves of Relu. 
elu converges faster to an optimal solution and in a better way compared to relu and elu does not have dying relu problem.
Due to which, elu performed better than relu.

Batch Normalization is another technique used. It applies normalization on outputs of activation functions from previous layers.
Due to normalization, the stability of network increases. Due to normalization, the parametric values lie in same range.
Which removes instability or roughness in the network which accelerates the process of learning
there by, reducing the required number of epochs and training time. It also acts as a regularizer, mimicing dropout.

"""

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

model.add(Conv2D(512, (3, 3), padding='same',  name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3), padding='same',  name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

model.add(Conv2D(4096, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(4096, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(10, (1, 1), padding='same'))
model.add(BatchNormalization())
"""
Used Global Average Pooling to convert the high dimension parameters to lower dimensions. It averages out pixels in a 2x2 window
It averages out spatial dimensions to 1.

"""
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
print(model.summary())

"""
I have used Nadam optimizer in the place of RMSProp.
Nadam essentially possesses the qualities of Adam and Nesterov Accelerated Gradient (NAG)
In Nadam,The learning process is accelerated by summing up the exponential decay of the moving averages for
the previous and current gradient.
Due to NAG feature, the NADAM has a controlled momentum which is aware of it's path towards achieving global minima.
We can yield better results with nadam.

Nadam is Adam + NAG and Adam is almost RMSProp + momentum. Therefore, nadam is an improvement on rmsprop and adam

"""
# initiate Nadam optimizer
opt = tf.keras.optimizers.Nadam(lr=0.0001, decay=1e-6)

# Let's train the model using Nadam
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:   # THIS CODE NOT TESTED IN TENSORFLOW 2.0.  IT IS AS IS!!!
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=0.,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.5)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch = steps_for_epoch,
                        epochs=epochs,
			callbacks=[LearningRateScheduler(lr_schedule)],
			
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



