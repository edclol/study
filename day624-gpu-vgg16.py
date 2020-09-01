import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16



# 使用全部GPU
strategy = tf.distribute.MirroredStrategy()

print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = VGG16(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      classifier_activation='softmax',)

model.summary()
model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose=1,
    callbacks=None,
    validation_split=0.,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False)

model.save(
    "filepath",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None)


