from pathlib import Path

import tensorflow as tf

from matplotlib import pyplot as plt

train_ds_fname = Path.cwd() / 'train.record'
test_ds_fname = Path.cwd() / 'test.record'

model_dir = Path.cwd() / 'model'

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

    #png = tf.io.decode_png(example['image/encoded'], channels=3)
    #png = tf.image.resize_with_crop_or_pad(png, 600, 960)

    hue = tf.io.decode_raw(example['image/histogram/hue'], tf.uint8) / 255
    hue = tf.ensure_shape(hue, (255,))
    #sat = tf.io.decode_raw(example['image/histogram/saturation'], tf.uint8) / 255
    #val = tf.io.decode_raw(example['image/histogram/value'], tf.uint8) / 255
    #out = tf.stack([hue, sat, val])

    return tf.stack([hue], axis=1), example['image/threshold'] / 255

def save_hist_graph(history, name='history.png'):
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(mae))

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mae, label='Training MAE')
    plt.plot(epochs_range, val_mae, label='Validation MAE')
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

for hist, thresh in train_ds.take(1):
    print(hist, thresh)

if model_dir.is_dir():
    model = tf.keras.models.load_model(str(model_dir))
else:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(255, 1)),
        tf.keras.layers.Conv1D(512, 16),
        tf.keras.layers.Conv1D(512, 16),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(1)
    ])

model.summary()

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

history = model.fit(train_ds, validation_data=test_ds, epochs=50)

model.save(model_dir)

save_hist_graph(history)
