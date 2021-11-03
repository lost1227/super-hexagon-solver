import win32gui
from mss import mss
import numpy as np
import cv2 as cv
import time
import keys

import game

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

record = False

show = True

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
            imgbw = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

            frame = game.GameFrame(imgbw, 20, 10, 20)
            if frame.is_valid():
                frame.findPath()
                move = frame.getNextMove()
                if move is None:
                    frame = game.GameFrame(imgbw, 10, 5, 1)
                    frame.findPath()
                    move = frame.getNextMove()

                if show or record:
                    plotted = frame.showPlottedPath()

                nextKey = None
                nextMove = frame.getNextMove()
                if nextMove == "LEFT":
                    nextKey = keys.VK_LEFT
                elif nextMove == "RIGHT":
                    nextKey = keys.VK_RIGHT
                
                if lastKey is not None:
                    keys.ReleaseKey(lastKey)
                lastKey = nextKey
                if nextKey is not None:
                    keys.PressKey(lastKey)
            elif show or record:
                plotted = cv.cvtColor(frame._thresh, cv.COLOR_GRAY2BGR)
                cv.rectangle(plotted, (5, 5), (plotted.shape[1]-5, plotted.shape[0]-5), (0, 0, 255), 10)

            if show or record:
                vis = np.concatenate((img, plotted), axis=0)
                # vis = cv.addWeighted(img, 0.6, plotted, 0.4, 0)

                cv.putText(vis, "{:.2f} FPS".format(fps), (5, window["height"]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            if show:
                cv.imshow("Hexagonical", vis)

            if record:
                out.write(vis)
            
            if show and cv.waitKey(1) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break

        if lastKey is not None:
            keys.ReleaseKey(lastKey)
    except KeyboardInterrupt:
        pass

    if record:
        out.release()