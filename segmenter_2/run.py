from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import cv2 as cv
from mss import mss
import time
import numpy as np

from windowinfo import find_window_coords

base_dir = Path.cwd()
model_dir = base_dir / 'model'

if not model_dir.exists():
    print("Model not found")
    exit(1)

tf.keras.backend.clear_session()
model = tf.keras.models.load_model(str(model_dir))

model.summary()

last_time = time.time()

def get_mask(img):
    resized_img = cv.resize(img, (960, 600))
    hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
    tf_image = tf.convert_to_tensor(hsv)
    tf_image = tf.cast(tf_image, tf.int32)
    tf_hue = tf_image[:, :, 0]
    tf_hue_hist = tfp.stats.histogram(tf_hue, edges=range(256))
    tf_hue_hist = tf_hue_hist / 255
    tf_input = tf.stack([tf_hue_hist], axis=1)
    tf_input = tf_input[tf.newaxis, ...]
    value = model(tf_input)

    value = value[0].numpy()
    value = np.clip(value, 0, 1)
    value = int(value * 255)

    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(mask, value, 255, cv.THRESH_BINARY)

    return value, mask

with mss() as sct:
    window = find_window_coords()

    try:
        while True:
            img = np.array(sct.grab(window))
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            width, height = img.shape[:2]

            fps = 1 / (time.time() - last_time)
            last_time = time.time()

            value, mask = get_mask(img)

            vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            cv.putText(vis, f"{value} ({fps:.2f} FPS)", (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv.imshow("Modeled", vis)

            key = cv.pollKey() & 0xFF

            if key == ord('q'):
                cv.destroyWindow("Modeled")
                break
    except KeyboardInterrupt:
        pass
