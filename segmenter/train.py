import numpy as np
import cv2 as cv
import os
import shutil

import pix2pix

import tensorflow as tf

from pathlib import Path

base_dir = Path.cwd()

val_split_ratio = 0.2
batch_size = 64
buffer_size = 100

data_dir = base_dir / 'data'
image_dir = data_dir / 'images'
mask_dir = data_dir / 'masks'

model_dir = base_dir / 'model'

epoch_ex_dir = base_dir / 'epoch_ends'

image_dims = (128, 128)

# https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control

image_paths = list(image_dir.glob("*.png"))
for path in image_paths:
    mask_path = mask_dir / path.name
    if not mask_path.is_file():
        print(f"Image {path.name} does not have a mask!")
        exit(1)

image_count = len(image_paths)

list_ds = tf.data.Dataset.list_files(str(image_dir / '*.png'))

def load_image(img):
    global image_dims
    img = tf.io.read_file(img)
    img = tf.io.decode_png(img)
    return tf.image.resize(img, image_dims)

def process_img(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img

def process_mask(mask):
    return tf.clip_by_value(mask, 0, 1)

def process_path(file_path):
    img = load_image(file_path)
    img_name = tf.strings.split(file_path, os.path.sep)[-1]
    mask_path = str(mask_dir) + os.path.sep + img_name
    mask = load_image(mask_path)
    return process_img(img), process_mask(mask)

full_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

val_size = int(image_count * val_split_ratio)
train_ds = full_ds.skip(val_size)
val_ds = full_ds.take(val_size)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

train_batches = (
    train_ds
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

test_batches = val_ds.batch(batch_size)

base_model = tf.keras.applications.MobileNetV2(input_shape=(image_dims+(3,)), include_top=False)
layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project'
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def display(imgs):
    for img in imgs:
        cv.imshow('display', img)
        while True:
            key = cv.waitKey() & 0xFF
            if key == 'n':
                break
            elif key == 'q':
                cv.destroyWindow('display')
                return
    cv.destroyWindow('display')

def show_predictions(dataset=None, num=1):
    imgs = []
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        pred_mask = create_mask(pred_mask)
        imgs.append(np.concatenate((image, mask, pred_mask), axis=1))
    display(imgs)

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if epoch_ex_dir.is_dir():
            shutil.rmtree(epoch_ex_dir)
        epoch_ex_dir.mkdir()

    def on_epoch_end(self, epoch, logs=None):
        data = next(test_batches.take(1).as_numpy_iterator())
        images, masks = data
        pred_masks = model.predict(images)

        imgs = []

        for i in range(min(5, len(images))):
            image = images[i]
            mask = masks[i]
            pred_mask = create_mask(pred_masks[i]).numpy()

            image = (image * 255).astype(np.uint8)
            mask = cv.cvtColor((mask * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
            pred_mask = cv.cvtColor((pred_mask * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
            img = np.concatenate((image, mask, pred_mask), axis=1)
            imgs.append(img)
        img = np.concatenate(imgs, axis=0)
        cv.imwrite(str(epoch_ex_dir / f"{epoch}.png"), img)

if model_dir.exists():
    model = tf.keras.models.load_model(str(model_dir))
else:
    model = unet_model(output_channels=2)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_batches, epochs=3, steps_per_epoch=image_count, validation_steps=val_size//batch_size//5, validation_data=test_batches, callbacks=[DisplayCallback()])

model.save(model_dir)

show_predictions(test_batches, 3)
