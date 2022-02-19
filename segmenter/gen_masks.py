import cv2 as cv
import numpy as np
from pathlib import Path
import shutil

base_dir = Path.cwd() / 'data'
in_dir = base_dir / 'images'
out_dir = base_dir / 'masks'

if not in_dir.is_dir():
    print("Input not found")
    exit(1)

if out_dir.exists() and not out_dir.is_dir():
    print("Output is not a directory")
    exit(1)

if not out_dir.exists():
    out_dir.mkdir()

dimensions = None
draw_mask = None
curr_image = None
processed_image = None
def mask_ui():
    height, width = dimensions
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

    processed_image = (processed_image * draw_mask).astype(np.uint8)

    vis = cv.cvtColor(processed_image, cv.COLOR_GRAY2BGR)
    vis = np.concatenate((curr_image, vis), axis=0)

    cv.imshow('image', vis)

lastx = lasty = 0
drawing = "none"
draw_thickness = 6

def mouse_callback(event, x, y, flags, param):
    global lastx, lasty, drawing, draw_mask, draw_thickness, dimensions

    y -= dimensions[0]

    if x < 0 or y < 0 or x >= dimensions[1] or y >= dimensions[0]:
        return

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = "primary"
        cv.circle(draw_mask, (x, y), draw_thickness, 0, cv.FILLED)
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = "secondary"
        cv.circle(draw_mask, (x, y), draw_thickness, 1, cv.FILLED)
    elif event in [cv.EVENT_MOUSEMOVE, cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
        if drawing == "primary":
            cv.line(draw_mask, (lastx, lasty), (x, y), 0, draw_thickness)
        elif drawing == "secondary":
            cv.line(draw_mask, (lastx, lasty), (x, y), 1, draw_thickness)
        if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
            drawing = "none"
    else:
        return
    lastx = x
    lasty = y
    redraw_threshold()

cv.namedWindow('image', cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
cv.createTrackbar('Threshold', 'image', 0, 255, redraw_threshold)
cv.setMouseCallback('image', mouse_callback)

files = list(in_dir.glob("*.png"))

i = 0
while i < len(files):
    inf = files[i]
    outf = out_dir / inf.name
    cv.setWindowTitle('image', inf.name)
    curr_image = cv.imread(str(inf), cv.IMREAD_COLOR)
    dimensions = curr_image.shape[:2]
    draw_mask = np.ones(dimensions)
    if curr_image is None:
        print(f"[{inf.name}] failed to read image")
        continue

    redraw_threshold()

    while cv.getWindowProperty('image', cv.WND_PROP_VISIBLE) >= 1:
        key = cv.waitKey(100) & 0xFF
        if key == ord('p'):
            exit()
        elif key == ord('d'):
            i += 1
            skip = False
            break
        elif key == ord('e'):
            i += 1
            skip = True
            break
        elif key == ord('a'):
            if i > 0:
                i -= 1
            skip = False
            break
        elif key == ord('q'):
            if i > 0:
                i -= 1
            skip = True
            break

    drawing = "none"

    if not skip:
        if outf.exists():
            outf.unlink()
        cv.imwrite(str(outf), processed_image)

cv.destroyWindow('image')
