import cv2 as cv
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from game import _threshmap as threshmap, _bins as bins

scriptdir = Path(__file__).resolve().parent

inpath = scriptdir / '../in'
if not inpath.is_dir():
    print("Input dir not found")
    exit(1)

outpath = scriptdir / '../out'
if outpath.is_dir():
    shutil.rmtree(outpath)
elif outpath.is_file():
    outpath.unlink()

outpath.mkdir()

interactive = False

def show_thresholded(img, gray, hueidx, threshval):
    _, thresh = cv.threshold(gray, threshval, 255, cv.THRESH_BINARY)
    bgrthresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    cv.putText(bgrthresh, "{} {}".format(hueidx, threshval), (10, bgrthresh.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    vis = np.concatenate((img, bgrthresh), axis=1)
    cv.imshow('thresh', vis)
try:
    for i, inf in enumerate(inpath.glob("*.png")):
        outf = outpath / inf.name
        img = cv.imread(str(inf), cv.IMREAD_COLOR)
        if img is None:
            print("Could not read image {}".format(inf))
            continue

        searchArea = img[150:445,330:630]

        hsv = cv.cvtColor(searchArea, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        huehist, _ = np.histogram(hsv[:,:,0], bins=bins)
        hueidx = huehist.argmax()

        bighuehist, _ = np.histogram(hsv[:,:,0], bins=range(181))
        hueval = bighuehist.argmax()

        threshval = threshmap[hueidx]
        if interactive:
            while True:
                show_thresholded(img, gray, hueidx, threshval)
                if cv.waitKey(0) & 0xFF != ord("e"):
                    break
                print('Enter threshold for {}'.format(hueidx))
                try:
                    threshval = int(input('> '))
                except ValueError:
                    pass
        
            threshmap[hueidx] = threshval

        gray = cv.blur(gray, (3, 3))
        _, thresh = cv.threshold(gray, threshval, 255, cv.THRESH_BINARY)

        bgrthresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

        cv.putText(bgrthresh, "{} -> {} -> {}".format(hueval, hueidx, threshval), (10, bgrthresh.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        print(hueidx, threshval)
        vis = np.concatenate((img, bgrthresh), axis=0)
        cv.imwrite(str(outf), vis)

        #plt.hist(flat, bins = range(256))
        #plt.show()
except KeyboardInterrupt:
    pass

cv.destroyAllWindows()
print("threshmap = [")
for i, val in enumerate(threshmap):
    print("    {}, # [{}, {})".format(val, i * 10, (i + 1) * 10))
print("]")
