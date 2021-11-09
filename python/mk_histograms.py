import cv2 as cv
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt

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

for i, inf in enumerate(inpath.glob("*.png")):
    #if i != 201:
    #    continue
    outf = outpath / inf.name
    img = cv.imread(str(inf), cv.IMREAD_COLOR)
    if img is None:
        print("Could not read image {}".format(inf))
        continue

    searchArea = img[150:445,330:630]

    hsv = cv.cvtColor(img[150:445,330:630], cv.COLOR_BGR2HSV)

    #huehist = np.histogram(hsv[:,:,0], bins=range(180))
    #sathist = np.histogram(hsv[:,:,1], bins=range(255))
    #valhist = np.histogram(hsv[:,:,2], bins=range(255))

    fig = plt.figure()
    fig.add_subplot(311)
    plt.hist(hsv[:,:,0].flatten(), bins=range(180))
    plt.xticks(range(0, 181, 10))
    fig.add_subplot(312)
    plt.hist(hsv[:,:,1].flatten(), bins=range(255))
    fig.add_subplot(313)
    plt.hist(hsv[:,:,2].flatten(), bins=range(255))
    
    fig.canvas.draw()

    graphs = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    graphs = graphs.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    if searchArea.shape[1] < graphs.shape[1]:
        padwidth = graphs.shape[1] - searchArea.shape[1]
        searchArea = np.pad(searchArea, ((0, 0), (padwidth//2, padwidth//2), (0, 0)), mode='constant', constant_values=255)
    else:
        padwidth = searchArea.shape[1] - graphs.shape[1]
        graphs = np.pad(graphs, ((0, 0), (padwidth//2, padwidth//2), (0, 0)), mode='constant', constant_values=255)
    
    vis = np.concatenate((searchArea, graphs), axis=0)
    cv.imwrite(str(outf), vis)

    #plt.hist(flat, bins = range(256))
    #plt.show()

    #if i > 10:
    #    exit()

