from pathlib import Path
import tensorflow as tf
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

last_time = time.time()

def get_mask(img):
    resized_img = cv.resize(img, (128, 128))
    normalized_img = resized_img.astype(np.float64) / 255.

    pred_mask = model.predict(np.expand_dims(normalized_img, axis=0))[0]
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = pred_mask.numpy()

    mask = (pred_mask * 255).astype(np.uint8)
    mask = cv.resize(mask, (img.shape[1], img.shape[0]))

    return mask

with mss() as sct:
    window = find_window_coords()

    try:
        while True:
            img = np.array(sct.grab(window))
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            width, height = img.shape[:2]

            fps = 1 / (time.time() - last_time)
            last_time = time.time()

            mask = get_mask(img)

            vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            cv.putText(vis, f"{fps:.2f} FPS", (5, height - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv.imshow("Modeled", vis)

            key = cv.pollKey() & 0xFF

            if key == ord('q'):
                cv.destroyWindow("Modeled")
                break
    except KeyboardInterrupt:
        pass
