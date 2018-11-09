from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import h5py

base_model = VGG16(input_tensor=Input((224, 224, 3)),
                   weights='imagenet', include_top=False)
# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights("./bottleneck_fc_model.h5")

# add the model on top of the convolutional base
model = Model(inputs=base_model.inputs, outputs=top_model(base_model.outputs))


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
  layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    classes=['daisy', 'dandelion', 'tulip', 'rose', 'sunflower'],
    class_mode="categorical",
    shuffle=False)

validation_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=32,
    classes=['daisy', 'dandelion', 'tulip', 'rose', 'sunflower'],
    class_mode="categorical",
    shuffle=False)

# fine-tune the model
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    epochs=50)
