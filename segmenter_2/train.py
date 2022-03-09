from pathlib import Path

import tensorflow as tf

from matplotlib import pyplot as plt

train_ds_fname = Path.cwd() / 'train.record'
test_ds_fname = Path.cwd() / 'test.record'

model_dir = Path.cwd() / 'model'

min_threshold = 40
max_threshold = 110
n_classes = (max_threshold - min_threshold) // 5

def tf_parse(eg):
    example = tf.io.parse_single_example(
        eg, {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/filename': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/threshold': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'image/histogram/hue': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/histogram/saturation': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/histogram/value': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        }
    )

    png = tf.io.decode_png(example['image/encoded'], channels=3)
    png = tf.image.resize_with_crop_or_pad(png, 600, 960)

    #hue = tf.io.decode_raw(example['image/histogram/hue'], tf.uint8) / 255
    #hue = tf.ensure_shape(hue, (255,))
    #sat = tf.io.decode_raw(example['image/histogram/saturation'], tf.uint8) / 255
    #val = tf.io.decode_raw(example['image/histogram/value'], tf.uint8) / 255
    #out = tf.stack([hue, sat, val])

    thresh = example['image/threshold']
    thresh = (thresh - min_threshold) / (max_threshold - min_threshold)
    thresh = thresh * n_classes
    thresh = tf.cast(thresh, tf.int32)

    return png, thresh

def save_hist_graph(history, name='history.png'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training MAE')
    plt.plot(epochs_range, val_acc, label='Validation MAE')
    plt.legend(loc='upper right')
    plt.title('Mean Absolute Error')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.savefig(name)

train_ds = tf.data.TFRecordDataset(filenames=[train_ds_fname])
test_ds = tf.data.TFRecordDataset(filenames=[test_ds_fname])

batch_size = 16

train_ds = train_ds.map(tf_parse).shuffle(100).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(tf_parse).shuffle(100).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(train_ds)

class ArgMax(tf.keras.layers.Layer):
    def __init__(self):
        super(ArgMax, self).__init__()

    def call(self, inputs):
        return tf.math.argmax(inputs, axis=-1) / inputs.shape[-1]

for hist, thresh in train_ds.take(1):
    print(hist, thresh)

if model_dir.is_dir():
    model = tf.keras.models.load_model(str(model_dir))
else:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(600, 960, 3)),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2, fill_mode='constant'),
        tf.keras.layers.Resizing(150, 240),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=test_ds, epochs=100)

model.save(model_dir)

save_hist_graph(history)
