from mss import mss
import numpy as np
import cv2 as cv
import time

import game
import shutil

from windowinfo import find_window_coords

from pathlib import Path

errordir = Path('./errors')

if errordir.is_dir():
    shutil.rmtree(errordir)
elif errordir.is_file():
    errordir.unlink()

errordir.mkdir()

last_time = 0

lastKey = None

record = True
show = False
keypress = True

if keypress:
    import pyautogui
    pyautogui.PAUSE = 0

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
                    nextKey = "left"
                elif move == "RIGHT":
                    nextKey = "right"
                
                if keypress and lastKey is not None:
                    pyautogui.keyUp(lastKey)
                lastKey = nextKey
                if keypress and nextKey is not None:
                    pyautogui.keyDown(lastKey)
            else:
                if show or record:
                    cv.imwrite(str(errordir / '{}.png'.format(i)), img)
                    i += 1
                    plotted = cv.cvtColor(frame._thresh, cv.COLOR_GRAY2BGR)
                    cv.rectangle(plotted, (5, 5), (plotted.shape[1]-5, plotted.shape[0]-5), (0, 0, 255), 10)
                if keypress and lastKey is not None:
                    pyautogui.keyUp(lastKey)
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

        if keypress and lastKey is not None:
            pyautogui.keyUp(lastKey)
    except KeyboardInterrupt:
        pass

    if record:
        out.release()
