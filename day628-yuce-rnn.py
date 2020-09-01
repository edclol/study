import os

data_dir = './'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

from matplotlib import pyplot as plt

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)

plt.plot(range(1440), temp[:1440])

# 数据标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# 生成时间序列样本及其目标的生成器
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# 准备训练生成器、验证生成器和测试生成器
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)


# 计算符合常识的基准方法的 MAE

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()

# 将 MAE 转换成摄氏温度误差
celsius_mae = 0.29 * std[1]

# 训练并评估一个密集连接模型
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# model.summary()
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# import matplotlib.pyplot as plt
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# 训练并评估一个基于 GRU 的模型
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# model.summary()
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# 训练并评估一个使用 dropout 正则化的基于 GRU 的模型

# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.2,
#                      recurrent_dropout=0.2,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# model.summary()
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# 训练并评估一个使用 dropout 正则化的堆叠 GRU 模型
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
model.summary()
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

# 使用逆序序列训练并评估一个 LSTM
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.datasets import imdb
#
# max_features = 10000
# maxlen = 500
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(
#     num_words=max_features)
#
# x_train = [x[::-1] for x in x_train]
# x_test = [x[::-1] for x in x_test]
#
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
#
# model = Sequential()
# model.add(layers.Embedding(max_features, 128))
# model.add(layers.LSTM(32))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
# model.summary()
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)

# 训练并评估一个双向 LSTM
#
# model = Sequential()
# model.add(layers.Embedding(max_features, 32))
# model.add(layers.Bidirectional(layers.LSTM(32)))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)

# 训练一个双向 GRU

model = Sequential()
model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
model.summary()
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
