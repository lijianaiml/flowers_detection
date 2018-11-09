from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# this is a PIL image
img = load_img('./train/dandelion/459633569_5ddf6bc116_m.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='./preview', save_prefix='rose', save_format='jpeg'):
  i += 1
  if i > 20:
    break  # otherwise the generator would loop indefinitely
