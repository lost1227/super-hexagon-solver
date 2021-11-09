import cv2 as cv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import shutil
import math

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

dr = 40
dtheta = 20

def pointDist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

for i, inf in enumerate(inpath.glob("*.png")):
    outf = outpath / inf.name
    img = cv.imread(str(inf), cv.IMREAD_GRAYSCALE)
    imgc = cv.imread(str(inf), cv.IMREAD_COLOR)
    if img is None or imgc is None:
        print("Could not read image {}".format(inf))
        continue

    dims = img.shape
    
    color = int(img[70:,:].max())
    
    cv.rectangle(img, (0, 0), (264, 40), color, -1)
    cv.rectangle(img, (dims[1]-350,0), (dims[1]-1,66), color, -1)

    img = cv.blur(img, (3,3))

    _, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    playerThresh = thresh[150:445,330:630]
    
    contours, _ = cv.findContours(playerThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    playerContour = None
    
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.005 * cv.arcLength(contour, True), True)
        
        area = cv.contourArea(approx)
        
        if contour[:,:,0].min() < 50 or contour[:,:,0].max() > (playerThresh.shape[1] - 50):
            continue
        if contour[:,:,1].min() < 50 or contour[:,:,1].max() > (playerThresh.shape[0] - 50):
            continue
        if area < 45 or area > 160:
            continue
        
        playerContour = approx + (330, 150)

    centerRadius = -1

    if playerContour is not None:
        center = (dims[1] / 2, dims[0] / 2)
        squeezedPlayer = np.squeeze(playerContour)
        closest = min(squeezedPlayer, key=lambda point: (point[0] - center[0])**2 + (point[1] - center[1])**2)
        
        centerRadius = pointDist(center, closest)
        cv.circle(thresh, np.int32(center), int(centerRadius), 255, -1)
    
    rgbthresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    
    if playerContour is not None:
        cv.drawContours(rgbthresh, [playerContour], 0, (0, 255, 255), -1)
        
        cornerDist = pointDist((0,0), center)
        
        for r in range(int(centerRadius), int(cornerDist), dr):
            for theta in range(0, 360, dtheta):
                x = int(center[0] - r * math.cos(math.radians(theta)))
                y = int(center[1] - r * math.sin(math.radians(theta)))
                
                if x < 0 or x >= dims[1]:
                    continue
                if y < 0 or y >= dims[0]:
                    continue

                if thresh[y,x] > 0:
                    continue

                cv.circle(rgbthresh, (x, y), 2, (255, 0, 255), -1)

    vis = np.concatenate((imgc, rgbthresh), axis=0)
    
    cv.imwrite(str(outf), vis)
