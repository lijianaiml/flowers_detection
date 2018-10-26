from keras import optimizers
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

base_model = ResNet50(input_tensor=Input((224, 224, 3)),
                      weights='imagenet', include_top=False)

for layers in base_model.layers[:-5]:
  layers.trainable = False


x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax')(x)
model = Model(base_model.input, x)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("compile done")

gen = ImageDataGenerator()
gen.flow_from_dataframe
train_generator = gen.flow_from_directory("train",
                                          (224, 224),
                                          shuffle=False,
                                          classes=['daisy', 'dandelion',
                                                   'tulip', 'rose',
                                                   'sunflower'],
                                          batch_size=16,
                                          class_mode="categorical")
test_generator = gen.flow_from_directory("test", (224, 224), shuffle=False,
                                         classes=['daisy', 'dandelion',
                                                  'tulip', 'rose',
                                                  'sunflower'],
                                         batch_size=16,
                                         class_mode="categorical")


def scheduler(epoch):
  if epoch % 20 == 0 and epoch != 0:
    lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, lr * 0.1)
    print("lr changed to %s" % (lr * 0.1))
  return K.get_value(model.optimizer.lr)


# reduce_lr = LearningRateScheduler(scheduler, verbose=1)
reduce_lr_onplat = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

tb_callback = TensorBoard(log_dir='./logs/eval-%d' % 2)

checkpointer = ModelCheckpoint(
    filepath='./tmp/current_model.h5', verbose=1, save_best_only=True, save_weights_only=False)


Callbacklist = [reduce_lr_onplat, tb_callback, checkpointer]

print(train_generator.samples)
print(train_generator.class_indices)
history = model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.samples /
                              train_generator.batch_size, validation_data=test_generator, validation_steps=test_generator.samples / test_generator.batch_size, epochs=8, callbacks=Callbacklist)
print("fit done")
# 保存模型
# model.save('resnet50_.h5')
# print("save .h5 file done")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
