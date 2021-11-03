import game
import cv2 as cv
from pathlib import Path
import shutil
import numpy as np

import time

scriptdir = Path(__file__).resolve().parent

inpath = scriptdir / 'in'
if not inpath.is_dir():
    print("Input dir not found")
    exit(1)

outpath = scriptdir / 'out'
if outpath.is_dir():
    shutil.rmtree(outpath)
elif outpath.is_file():
    outpath.unlink()

outpath.mkdir()

for i, inf in enumerate(inpath.glob("*.png")):
    start = time.time()
    outf = outpath / inf.name
    img = cv.imread(str(inf), cv.IMREAD_COLOR)
    if img is None:
        print("Could not read image {}".format(inf))
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    try:
        frame = game.GameFrame(gray)
    except ValueError:
        continue
    
    frame.findPath()

    plotted = frame.showPlottedPath()

    vis = np.concatenate((img, plotted), axis=0)
    cv.imwrite(str(outf), vis)
    
    diff = time.time() - start

    print("Frame {} took {:02f}s ({:02f} fps)".format(i, diff, 1/diff))
