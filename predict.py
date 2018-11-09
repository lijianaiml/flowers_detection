from keras.models import *
from keras.layers import *
import keras
import numpy as np
import h5py
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import *

model = load_model('model.h5')

X_test = []
for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
  with h5py.File(filename, 'r') as h:
    X_test.append(np.array(h['test']))
X_test = np.concatenate(X_test, axis=1)

y_pred = model.predict(X_test, verbose=1)

predict = backend.argmax(y_pred, axis=1)
predict2 = backend.max(y_pred, axis=1)

with tf.Session() as sess:
  result_class = sess.run(predict)
  result_confidence = sess.run(predict2)


df = pd.read_csv("./sample_submission.csv")

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("test", (224, 224), shuffle=False,
                                         batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
  df.set_value(i, 'label', result_class[i])
  df.set_value(i, 'fname', fname)
  df.set_value(i, 'confidence', result_confidence[i])


df.to_csv('pred_test.csv', index=False)
df.head(10)

# print(df.head(10))
