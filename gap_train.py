import h5py
import numpy as np
from sklearn.utils import shuffle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.utils import plot_model

# 定义matplotlib可视化类


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))

        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# 定义tensorboard回调函数
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                         batch_size=16, write_graph=True, write_grads=False,
                                         write_images=True)

np.random.seed(2017)

X_train = []
X_test = []

for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])


X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)


X_train, y_train = shuffle(X_train, y_train)
y_train = to_categorical(y_train, 5)


input_tensor = Input(X_train.shape[1:])
print(input_tensor.shape)
x = Dropout(0.6)(input_tensor)
x = Dense(5, activation='softmax')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("compile done")

history = LossHistory()

model.fit(X_train, y_train, batch_size=128,
          epochs=8, validation_split=0.2, callbacks=[history])
print("fit done")

# 使用matplotlib画loss和acc曲线
history.loss_plot('epoch')


# 打印模型
# model.summary()

#
# plot_model(model, to_file='./model.png')

# 保存模型
# model.save('model.h5')
# print("save .h5 file done")
