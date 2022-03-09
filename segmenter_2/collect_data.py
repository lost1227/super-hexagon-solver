from pathlib import Path
import cv2 as cv
from mss import mss
from datetime import datetime
import numpy as np

from windowinfo import find_window_coords

out_dir = Path.cwd() / 'data'

if not out_dir.exists():
    out_dir.mkdir()

with mss() as sct:
    window = find_window_coords()

    try:
        while True:
            img = np.array(sct.grab(window))
            width, height = img.shape[:2]

            out_f = out_dir / f'{datetime.now().strftime("%m%d_%H%M%S_%f")}.png'

            cv.imshow("Data", img)
            cv.imwrite(str(out_f), img)

            key = cv.waitKey(500) & 0xFF

            if key == ord('q'):
                cv.destroyWindow("Data")
                break
    except KeyboardInterrupt:
        pass
