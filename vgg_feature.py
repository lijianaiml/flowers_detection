from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

base_model = VGG16(input_tensor=Input((224, 224, 3)),
                   weights='imagenet', include_top=False)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    # this means our generator will only yield batches of data, no labels
    classes=['daisy', 'dandelion', 'tulip', 'rose', 'sunflower'],
    class_mode="categorical",
    shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data
bottleneck_features_train = base_model.predict_generator(train_generator)
# save the output as a Numpy array
np.save(open('./feature/bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)
np.save(open('./feature/bottleneck_features_train_label.npy', 'wb'),
        train_generator.classes)

test_generator = datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=32,
    classes=['daisy', 'dandelion', 'tulip', 'rose', 'sunflower'],
    class_mode="categorical",
    shuffle=False)
bottleneck_features_validation = base_model.predict_generator(test_generator)
np.save(open('./feature/bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)
np.save(open('./feature/bottleneck_features_validation_label.npy', 'wb'),
        test_generator.classes)
