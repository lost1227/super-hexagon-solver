import cv2 as cv
import numpy as np
import math

def _point_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class GameFrame():
    too_close = 5
    dr = 40
    dtheta = 20

    def __init__(self, frame):
        if frame is None:
            raise ValueError("No frame provided")
        self._frame = frame
        self._dims = frame.shape
        self._center = (self._dims[1] / 2, self._dims[0] / 2)

        self._cover_ui()
        self._threshold()
        self._find_player_contour()
        if self._player_contour is None:
            raise ValueError("Could not locate player")
        self._cover_center()
        self._setup_search_grid()

    def _cover_ui(self):
    
        color = int(self._frame[70:,:].max())
    
        cv.rectangle(self._frame, (0, 0), (264, 40), color, -1)
        cv.rectangle(self._frame, (self._dims[1]-350,0), (self._dims[1]-1,66), color, -1)

    def _threshold(self):
        frame = cv.blur(self._frame, (3,3))
        _, self._thresh = cv.threshold(self._frame, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    def _find_player_contour(self):
        playerThresh = self._thresh[150:445,330:630]
        contours, _ = cv.findContours(playerThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        self._player_contour = None
        
        for contour in contours:
            approx = cv.approxPolyDP(contour, 0.005 * cv.arcLength(contour, True), True)

            area = cv.contourArea(approx)

            if contour[:,:,0].min() < 50 or contour[:,:,0].max() > (playerThresh.shape[1] - 50):
                continue
            if contour[:,:,1].min() < 50 or contour[:,:,1].max() > (playerThresh.shape[0] - 50):
                continue
            if area < 45 or area > 160:
                continue

            self._player_contour = approx + (330, 150)
    
    def _cover_center(self):
        squeezedPlayer = np.squeeze(self._player_contour)
        self._closest = min(squeezedPlayer, key=lambda point: (point[0] - self._center[0])**2 + (point[1] - self._center[1])**2)
        
        self._center_radius = _point_dist(self._center, self._closest)
        cv.circle(self._thresh, np.int32(self._center), int(self._center_radius - GameFrame.too_close - 1), 255, -1)
        
    def _setup_search_grid(self):
        corner_dist = _point_dist((0,0), self._center)
        self._grid = np.zeros( ( int((corner_dist - self._center_radius) / GameFrame.dr), int(360 / GameFrame.dtheta)), np.int32 )
        
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):            
                r = (GameFrame.dr * y) + self._center_radius
                theta = GameFrame.dtheta * x
                
                realx = int(self._center[0] + r * math.cos(math.radians(theta)))
                realy = int(self._center[1] - r * math.sin(math.radians(theta)))
                
                if realx < 0 or realx >= self._dims[1]:
                    continue
                if realy < 0 or realy >= self._dims[0]:
                    continue

                if y > 0:
                    area_around = self._thresh[max(0, realy-GameFrame.too_close):min(self._dims[0], realy+GameFrame.too_close),max(0, realx-GameFrame.too_close):min(self._dims[1], realx+GameFrame.too_close)]
                    if area_around.max() > 0:
                        continue
                self._grid[y][x] = 1
                
    def showPlottedPath(self):
        rgbthresh = cv.cvtColor(self._thresh, cv.COLOR_GRAY2BGR)
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):
                if not self._grid[y][x] == 1:
                    continue

                r = (GameFrame.dr * y) + self._center_radius
                theta = GameFrame.dtheta * x
                
                realx = int(self._center[0] + r * math.cos(math.radians(theta)))
                realy = int(self._center[1] - r * math.sin(math.radians(theta)))
                
                if x == 0 and y == 0:
                    cv.circle(rgbthresh, (realx, realy), 2, (0, 0, 255), -1)
                else:
                    cv.circle(rgbthresh, (realx, realy), 2, (255, 0, 255), -1)
        
        cv.drawContours(rgbthresh, [self._player_contour], 0, (0, 255, 0), -1)
        return rgbthresh
