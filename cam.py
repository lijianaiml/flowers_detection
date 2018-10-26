from keras.models import *
from keras.layers import *
import keras
import numpy as np
import h5py
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import *
import matplotlib
import matplotlib.pyplot as plt
import random
import cv2
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

print(matplotlib.get_backend())
print(matplotlib.matplotlib_fname())

model = load_model('./resnet50_.h5')
layer_list = list(zip([x.name for x in model.layers], range(len(model.layers))))
# print(zip([x.name for x in model.layers], range(len(model.layers))))
# print(layer_list)


weights = model.layers[177].get_weights()[0]
model2 = Model(model.input, [model.layers[172].output, model.output])


plt.figure(figsize=(12, 14))
for i in range(16):
  plt.subplot(4, 4, i + 1)
  img = cv2.imread('./test/test/%s.jpg' % (i + 1))
  img = cv2.resize(img, (224, 224))
  x = img.copy()
  x.astype(np.float32)
  out, prediction = model2.predict(np.expand_dims(x, axis=0))

  # predict = tf.argmax(y_pred,axis=1)
  predict = backend.argmax(prediction, axis=1)
  predict2 = backend.max(prediction, axis=1)
  # predict2 = tf.decode_predictions(y_pred, top=top_n)[0]

  with tf.Session() as sess:
    result_class = sess.run(predict)
    result_confidence = sess.run(predict2)

  print("++++++++++++++++")
  print(out.shape)
  print(weights.shape)
  print(prediction.shape)

  prediction = prediction[0]
  out = out[0]
  plt.title('%d %.2f%%' % (result_class, result_confidence * 100))
  # if prediction < 0.5:
  #   plt.title('cat %.2f%%' % (100 - prediction * 100))
  # else:
  #   plt.title('dog %.2f%%' % (prediction * 100))

  # cam = (prediction - 0.5) * np.matmul(out, weights)
  cam = np.matmul(out, weights[:, result_class])
  cam -= cam.min()
  cam /= cam.max()
  cam -= 0.2
  cam /= 0.8

  cam = cv2.resize(cam, (224, 224))
  heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
  heatmap[np.where(cam <= 0.15)] = 0

  print(heatmap.shape)

  out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)

  plt.axis('off')
  plt.imshow(out[:, :, ::-1])
plt.show()
