import cv2 as cv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import shutil

import code

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
    outf = outpath / inf.name
    img = cv.imread(str(inf), cv.IMREAD_GRAYSCALE)
    imgc = cv.imread(str(inf), cv.IMREAD_COLOR)
    if img is None or imgc is None:
        print("Could not read image {}".format(inf))
        continue

    img = img[66:,:]
    
    img = cv.blur(img, (3,3))
    
    _, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    rgbthresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
 
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.005 * cv.arcLength(contour, True), True)
        
        area = cv.contourArea(approx)
        
        if area > 50 and area < 160:
            cv.drawContours(rgbthresh, [approx], 0, (0, 255, 255), -1)
        else:
            cv.drawContours(rgbthresh, [approx], 0, (0, 255, 0), 3)
        
            for point in np.squeeze(approx):
                cv.circle(rgbthresh, point, 2, (255, 0, 255), -1)
        
        if inf.name == "frame-1-0010.png":
            print(area)
        
        # code.interact(local=locals())
    
    vis = np.concatenate((imgc, rgbthresh), axis=0)
    
    cv.imwrite(str(outf), vis)
    
    #if i > 10:
    #    exit()
