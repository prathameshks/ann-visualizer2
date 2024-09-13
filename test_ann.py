from ann_visualizer2.visualize import ann_viz

import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D, Dropout,Flatten
model2 = Sequential()

# add all types of layers in the model

model2.add(Conv2D(8, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(1, activation='softmax'))

ann_viz(model2)