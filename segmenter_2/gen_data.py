import cv2 as cv
from cv2 import getTrackbarPos
import numpy as np
from pathlib import Path

import tensorflow as tf

img_dir = Path.cwd() / 'data'
out_f = Path.cwd() / 'data.record'

if not img_dir.is_dir():
    print("Input not found")
    exit(1)

thresholds = {}
files = list(img_dir.glob("*.png"))

if out_f.exists():
    raw_dataset = tf.data.TFRecordDataset([out_f])
    for record in raw_dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(record)
        features = example.features.feature
        name = features['image/filename'].bytes_list.value[0].decode('utf-8')
        threshold = features['image/threshold'].int64_list.value[0]

        path = img_dir / name
        thresholds[path] = threshold


dimensions = None
curr_image = None
processed_image = None
def mask_ui():
    _, width = dimensions
    left_ui_width = 0
    while (curr_image[0, left_ui_width] <= (30, 30, 30)).all():
        left_ui_width += 1

    right_ui_width = 0
    while (curr_image[60, width - right_ui_width - 1] <= (30, 30, 30)).all():
        right_ui_width += 1

    left_ui_poly = np.array([
        (0, 0), (left_ui_width, 0), (left_ui_width - 29, 40), (0, 40)
    ])
    right_ui_poly = np.array([
        (width - (right_ui_width + 164), 0), (width, 0), (width, 65),
        (width - (right_ui_width - 2), 65), (width - (right_ui_width + 12), 40),
        (width - (right_ui_width + 135), 40)
    ])

    cv.fillPoly(processed_image, [left_ui_poly], color=(255, 255, 255))
    cv.fillPoly(processed_image, [right_ui_poly], color=(255, 255, 255))

def redraw_threshold(x = None):
    global curr_image, processed_image, draw_mask, dimensions
    if x is None:
        x = cv.getTrackbarPos('Threshold', 'image')

    processed_image = cv.cvtColor(curr_image, cv.COLOR_BGR2GRAY)

    mask_ui()

    _, processed_image = cv.threshold(processed_image, x, 255, cv.THRESH_BINARY)

    vis = cv.cvtColor(processed_image, cv.COLOR_GRAY2BGR)
    vis = np.concatenate((curr_image, vis), axis=1)

    cv.imshow('image', vis)

cv.namedWindow('image', cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
cv.createTrackbar('Threshold', 'image', 0, 255, redraw_threshold)

i = 0
quit = False
while i < len(files) and not quit:
    inf = files[i]
    cv.setWindowTitle('image', inf.name)
    curr_image = cv.imread(str(inf), cv.IMREAD_COLOR)
    if curr_image is None:
        print(f"[{inf.name}] failed to read image")
        continue
    dimensions = curr_image.shape[:2]

    if inf in thresholds:
        cv.setTrackbarPos('Threshold', 'image', thresholds[inf])

    redraw_threshold()

    while cv.getWindowProperty('image', cv.WND_PROP_VISIBLE) >= 1:
        key = cv.waitKey(100) & 0xFF
        if key == ord('p'):
            quit = True
            break
        elif key == ord('d'):
            i += 1
            break
        elif key == ord('a'):
            if i > 0:
                i -= 1
            break

    thresholds[inf] = cv.getTrackbarPos('Threshold', 'image')

    if cv.getWindowProperty('image', cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyWindow('image')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

with tf.io.TFRecordWriter(str(out_f)) as writer:
    for f in files:
        if f not in thresholds:
            continue

        img = cv.imread(str(inf), cv.IMREAD_COLOR)
        if img is None:
            continue
        success, encoded = cv.imencode(".png", img)
        if not success:
            continue

        writer.write(tf.train.Example(features=tf.train.Features(feature={
            "image/filename": _bytes_feature(f.name.encode('utf-8')),
            "image/threshold": _int64_feature(thresholds[f]),
            "image/encoded": _bytes_feature(bytes(encoded))
        })).SerializeToString())
