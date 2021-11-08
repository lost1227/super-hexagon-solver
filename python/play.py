import win32gui
from mss import mss
import numpy as np
import cv2 as cv
import time
import keys

import game
import shutil

from pathlib import Path

errordir = Path('./errors')

if errordir.is_dir():
    shutil.rmtree(errordir)
elif errordir.is_file():
    errordir.unlink()

errordir.mkdir()

def find_window_coords():
    coords = None
    def callback(hwnd, extra):
        nonlocal coords
        text = win32gui.GetWindowText(hwnd)
        if text != "Super Hexagon":
            return
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0] + 10
        y = rect[1] + 40
        w = rect[2] - 10 - x
        h = rect[3] - 50 - y
        coords = {"top":y, "left":x, "width":w, "height":h}
    win32gui.EnumWindows(callback, None)

    return coords

last_time = 0

lastKey = None

record = True
show = True

i = 0

with mss() as sct:
    window = find_window_coords()

    if window is None:
        print("Could not find window!")
        exit(1)


    if record:
        out = cv.VideoWriter('out.mp4',cv.VideoWriter_fourcc('M','P','4','V'), 10, (window["width"],window["height"]*2))

    try:
        while True:
            img = np.array(sct.grab(window))

            fps = 1 / (time.time() - last_time)
            last_time = time.time()
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

            frame = game.GameFrame(img)
            if frame.is_valid():
                move = frame.getNextMove()

                if show or record:
                    plotted = frame.showPlottedPath()

                nextKey = None
                if move == "LEFT":
                    nextKey = keys.VK_LEFT
                elif move == "RIGHT":
                    nextKey = keys.VK_RIGHT
                
                if lastKey is not None:
                    keys.ReleaseKey(lastKey)
                lastKey = nextKey
                if nextKey is not None:
                    keys.PressKey(lastKey)
            else:
                if show or record:
                    cv.imwrite(str(errordir / '{}.png'.format(i)), img)
                    i += 1
                    plotted = cv.cvtColor(frame._thresh, cv.COLOR_GRAY2BGR)
                    cv.rectangle(plotted, (5, 5), (plotted.shape[1]-5, plotted.shape[0]-5), (0, 0, 255), 10)
                if lastKey is not None:
                    keys.ReleaseKey(lastKey)
                lastKey = None
                

            if show or record:
                vis = np.concatenate((img, plotted), axis=0)
                # vis = cv.addWeighted(img, 0.6, plotted, 0.4, 0)

                cv.putText(vis, "{:.2f} FPS".format(fps), (5, window["height"]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            if show:
                cv.imshow("Hexagonical", vis)

            if record:
                out.write(vis)
            
            if show and cv.pollKey() & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break

        if lastKey is not None:
            keys.ReleaseKey(lastKey)
    except KeyboardInterrupt:
        pass

    if record:
        out.release()