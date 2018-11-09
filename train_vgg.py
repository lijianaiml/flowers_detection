
import numpy as np

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import h5py
from keras.utils import to_categorical


train_data = np.load(open('./feature/bottleneck_features_train.npy', 'rb'))
train_labels = np.load(
    open('./feature/bottleneck_features_train_label.npy', 'rb'))
train_labels = to_categorical(train_labels, 5)

validation_data = np.load(
    open('./feature/bottleneck_features_validation.npy', 'rb'))
validation_labels = np.load(
    open('./feature/bottleneck_features_validation_label.npy', 'rb'))
validation_labels = to_categorical(validation_labels, 5)

print('++++++++++++')
print(train_data.shape)
print(train_labels.shape)
print(train_labels)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          nb_epoch=50, batch_size=32,
          validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')
