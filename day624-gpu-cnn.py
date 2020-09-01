

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.datasets import mnist
# from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
# import cv2  # 加载opencv，为了后期的图像处理
# from tensorflow.keras import datasets
# import h5py as h5py
import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.utils import plot_model
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import ModelCheckpoint
ishape = 64
def get_compiled_model():
    ishape = 64
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(ishape, ishape, 3))
    for layers in model_vgg.layers:
        layers.trainable = False

    model = Flatten()(model_vgg.output)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dropout(0.5)(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation='softmax', name='prediction')(model)
    # model = multi_gpu_model(model, gpus=2)
    model_vgg_cifar10_pretrain = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
    model_vgg_cifar10_pretrain.summary()
    model_vgg_cifar10_pretrain.compile(optimizer='sgd', loss='categorical_crossentropy',
                                       metrics=['accuracy'])

    return model_vgg_cifar10_pretrain







# Train the model on all available devices.
ishape = 64
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = [tf.image.resize(i, (ishape, ishape)) for i in X_train]
X_test = [tf.image.resize(i, (ishape, ishape)) for i in X_test]

# X_train = [cv2.resize(i, (ishape, ishape)) for i in X_train]
# X_test = [cv2.resize(i, (ishape, ishape)) for i in X_test]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
print(X_train[0].shape)
print(y_train[0])

X_train = X_train / 255
X_test = X_test / 255

np.where(X_train[0] != 0)


# 哑编码
def train_y(y):
    y_one = np.zeros(10)
    y_one[y] = 1
    return y_one


y_train_one = np.array([train_y(y_train[i]) for i in range(len(y_train))])
y_test_one = np.array([train_y(y_test[i]) for i in range(len(y_test))])

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()
# 迭代训练这个地方要加入callbacks
model.fit(X_train, y_train_one,validation_data=(X_test, y_test_one),
                                         epochs=5000, batch_size=128,
                                         validation_split=0.1,
                                         verbose=1,
                                         shuffle=True)


