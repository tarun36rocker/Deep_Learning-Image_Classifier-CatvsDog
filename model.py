from keras.models import Sequential #initialise the neural network as its a sequence of layers
from keras.layers import Conv2D #convolutional layers
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense #add the layers to the ann
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) #32 is number of feature detectors having dimensions (3,3),input_shape is format into which images will be converted .. 3 channels (rgb) of 64 * 64 coloured pixels, relu function for non-colinearity

# Step 2 - Pooling .. reducing size of the feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2))) #2,2 is used so that we dont loose information and also be precise in where features are detected

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) #no. of nodes = 128 ( in power of 2 ), number around 100 is good 
classifier.add(Dense(units = 1, activation = 'sigmoid')) # binary output

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # adam s stochastic gradient descent algorithm 

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator #preprocesses images to prevent overfitting and enrichs our training set without increasing the number of images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1, #reduce number of epochs because c;u times out
                         validation_data = test_set,
                         validation_steps = 2000)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
from keras.preprocessing import image
model = pickle.load(open('model.pkl','rb'))
test_image=image.load_img('dataset/single_prediction/snoomy.jpg',target_size = (64, 64))
test_image=image.img_to_array(test_image) #converts it into 3d array
test_image=np.expand_dims(test_image,axis=0)
print(model.predict([test_image]))