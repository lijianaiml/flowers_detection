from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py


def write_gap(CNN_MODEL, image_size, lambda_func=None):
  print(CNN_MODEL.__name__)
  width = image_size[0]
  height = image_size[1]
  input_tensor = Input((height, width, 3))
  x = input_tensor
  if lambda_func:
    x = Lambda(lambda_func)(x)
  base_model = CNN_MODEL(
      input_tensor=x, weights='imagenet', include_top=False)
  model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

  gen = ImageDataGenerator()
  train_generator = gen.flow_from_directory("train",
                                            image_size,
                                            shuffle=False,
                                            classes=['daisy', 'dandelion',
                                                     'tulip', 'rose',
                                                     'sunflower'],
                                            batch_size=16,
                                            class_mode="binary")
  test_generator = gen.flow_from_directory("test", image_size, shuffle=False,
                                           batch_size=16, class_mode=None)

  train = model.predict_generator(
      train_generator, train_generator.samples / train_generator.batch_size)
  print("train data is done")
  test = model.predict_generator(
      test_generator, test_generator.samples / test_generator.batch_size)
  print("test data is done")
  with h5py.File("gap_%s.h5" % CNN_MODEL.__name__) as h:
    h.create_dataset("train", data=train)
    h.create_dataset("test", data=test)
    h.create_dataset("label", data=train_generator.classes)



# return
# write_gap(resnet50.ResNet50.__name__, (224, 224))
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)
write_gap(VGG16, (224, 224))
write_gap(VGG19, (224, 224))
