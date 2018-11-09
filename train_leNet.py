from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import h5py


model = Sequential()
model.add(Conv2D(32, (3, 3),
                 input_shape=(224, 224, 3),
                 kernel_initializer='he_normal',
                 bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),
                 kernel_initializer='he_normal',
                 bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),
                 kernel_initializer='he_normal',
                 bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64,
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])


train_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_gen.flow_from_directory(
    'train',
    target_size=(224, 224),
    classes=['daisy', 'dandelion', 'tulip', 'rose', 'sunflower'],
    batch_size=32,
    class_mode="categorical")

validation_generator = test_gen.flow_from_directory(
    'test',
    target_size=(224, 224),
    classes=['daisy', 'dandelion', 'tulip', 'rose', 'sunflower'],
    batch_size=32,
    class_mode="categorical")

# reduce_lr = LearningRateScheduler(scheduler, verbose=1)
reduce_lr_onplat = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

tb_callback = TensorBoard(log_dir='./logs/first/%d' % 1)

checkpointer = ModelCheckpoint(
    filepath='./tmp/first/current_model.h5', verbose=0, save_best_only=True, save_weights_only=False)


Callbacklist = [reduce_lr_onplat, tb_callback, checkpointer]

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    epochs=50,
    callbacks=Callbacklist)

# always save your weights after training or during training
# model.save_weights('first_try.h5')
