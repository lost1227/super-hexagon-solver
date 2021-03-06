import cv2 as cv
import numpy as np
import math
import heapq
import time

_threshmap = [
    58, # [0, 7)
    70, # [7, 10)
    80, # [10, 15)
    85, # [15, 20)
    90, # [20, 30)
    80, # [30, 40)
    80, # [40, 50)
    80, # [50, 60)
    80, # [60, 70)
    80, # [70, 80)
    80, # [80, 90)
    80, # [90, 100)
    70, # [100, 110)
    69, # [110, 120)
    63, # [120, 130)
    68, # [130, 140)
    80, # [140, 150)
    80, # [150, 160)
    80, # [160, 170)
    62, # [170, 180]
]

_bins = [0, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

_debug_contours = False

def _point_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class AStar_Node():
    def __init__(self, point, prev, pathlen, totalcost):
        self.point = point
        self.prev = prev
        self.pathlen = pathlen
        self.totalcost = totalcost

    def __lt__(self, other):
        return self.totalcost < other.totalcost

class _GridParams:
    def __init__(self, dr, dtheta, too_close):
        self.dr = dr
        self.dtheta = dtheta
        self.too_close = too_close

class GameFrame():
    _GRID_OPEN = 4
    _GRID_OOB = 2
    _GRID_BLOCKED = 1

    _gridParams = [
        _GridParams(25, 10, 15),
        #_GridParams(20, 8, 15),
        #_GridParams(10, 8, 15),
        _GridParams(10, 8, 5)
    ]

    _maxPathTime = 0.1

    def __init__(self, frame):
        self._colorframe = frame
        self._frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self._dims = frame.shape
        self._center = (self._dims[1] / 2, self._dims[0] / 2)

        self._path = []

        self._cover_ui()
        self._threshold()
        self._find_player_contour()
        if self._player_contour is None:
            return
        self._cover_center()
        for i, param in enumerate(GameFrame._gridParams):
            self._gridPrecisionLevel = i
            self._setup_search_grid(param.dr, param.dtheta, param.too_close)
            self._find_path()
            if len(self._path) > 0:
                break

    def is_valid(self):
        return self._player_contour is not None

    def _cover_ui(self):
    
        #color = int(self._frame[70:,:].max())
        color = 255
    
        cv.rectangle(self._frame, (0, 0), (264, 40), color, -1)
        cv.rectangle(self._frame, (self._dims[1]-350,0), (self._dims[1]-1,66), color, -1)

    def _threshold(self):
        threshOrigin = (int(self._center[1]-147.5), int(self._center[0]-150))
        searchArea = self._colorframe[threshOrigin[0]:threshOrigin[0]+295,threshOrigin[1]:threshOrigin[1]+300]
        hsv = cv.cvtColor(searchArea, cv.COLOR_BGR2HSV)
        huehist, _ = np.histogram(hsv[:,:,0], bins=_bins)
        hueidx = huehist.argmax()
        threshval = _threshmap[hueidx]
        self._debugThreshInfo = (hueidx, threshval)
        #_, self._thresh = cv.threshold(self._frame, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        blurframe = cv.blur(self._frame, (3, 3))
        _, self._thresh = cv.threshold(blurframe, threshval, 255, cv.THRESH_BINARY)

    def _find_player_contour(self):
        playerThreshOrigin = (int(self._center[1]-147.5), int(self._center[0]-150))
        playerThresh = self._thresh[playerThreshOrigin[0]:playerThreshOrigin[0]+295,playerThreshOrigin[1]:playerThreshOrigin[1]+300]
        contours, _ = cv.findContours(playerThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        self._player_contour = None
        
        player_candidates = []

        for contour in contours:
            approx = cv.approxPolyDP(contour, 0.005 * cv.arcLength(contour, True), True)

            area = cv.contourArea(approx)
            
            if _debug_contours:
                print(area)
                copy = cv.cvtColor(playerThresh, cv.COLOR_GRAY2BGR)
                cv.drawContours(copy, [approx], 0, (0, 255, 255), -1)
                cv.imshow('Contour', copy)
                cv.waitKey(0)
                cv.destroyWindow('Contour')

            if contour[:,:,0].min() < 25 or contour[:,:,0].max() > (playerThresh.shape[1] - 25):
                continue
            if contour[:,:,1].min() < 25 or contour[:,:,1].max() > (playerThresh.shape[0] - 25):
                continue
            if area < 2 or area > 160:
                continue

            player_candidates.append((approx + (playerThreshOrigin[1], playerThreshOrigin[0]), area))
        if len(player_candidates) > 0:
            self._player_contour = max(player_candidates, key=lambda pair: pair[1])[0]
        if self._player_contour is not None:
            boundRect = cv.boundingRect(self._player_contour)
            cv.rectangle(self._thresh, boundRect, 0, -1)
    
    def _cover_center(self):
        assert len(self._player_contour) > 0
        squeezedPlayer = np.squeeze(self._player_contour)
        self._closest = min(squeezedPlayer, key=lambda point: (point[0] - self._center[0])**2 + (point[1] - self._center[1])**2)
        
        self._center_radius = _point_dist(self._center, self._closest)
        cv.circle(self._thresh, np.int32(self._center), int(self._center_radius), 0, -1)
        
    def _setup_search_grid(self, dr, dtheta, too_close):
        corner_dist = _point_dist((0,0), self._center)
        self._grid = np.zeros( ( int((corner_dist - self._center_radius) / dr), int(360 / dtheta)), np.int32 )
        self._real_coords = np.zeros( (self._grid.shape[0], self._grid.shape[1], 2), np.int32)

        player_vector = (self._closest[0] - self._center[0], self._closest[1] - self._center[1])

        if player_vector[0] == 0:
            player_theta = math.pi / 2 if player_vector[1] > 0 else -1 *math.pi / 2
        else:
            player_theta = math.atan(player_vector[1] / player_vector[0])

        if player_vector[0] < 0:
            player_theta += math.pi
        player_theta = math.degrees(player_theta)

        self._player_theta = player_theta
        
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):            
                r = (dr * y) + self._center_radius
                theta = (dtheta * x) - self._player_theta
                
                realx = int(self._center[0] + r * math.cos(math.radians(theta)))
                realy = int(self._center[1] - r * math.sin(math.radians(theta)))

                self._real_coords[y, x] = (realx, realy)
                
                if realx < 0 or realx >= self._dims[1]:
                    self._grid[y][x] = GameFrame._GRID_OOB
                    continue
                if realy < 0 or realy >= self._dims[0]:
                    self._grid[y][x] = GameFrame._GRID_OOB
                    continue

                area_around = self._thresh[max(0, realy-too_close):min(self._dims[0], realy+too_close),max(0, realx-too_close):min(self._dims[1], realx+too_close)]
                if area_around.any():
                    self._grid[y][x] = GameFrame._GRID_BLOCKED
                    continue

                if (dr * y < 80):
                    fail = False
                    for i in range(40):
                        testx = int(realx + i * math.cos(math.radians(theta)))
                        testy = int(realy - i * math.sin(math.radians(theta)))
                        if testx < 0 or testx >= self._dims[1]:
                            break
                        if testy < 0 or testy >= self._dims[0]:
                            break
                        if self._thresh[testy, testx] > 0:
                            fail = True
                            break
                    if fail:
                        continue

                self._grid[y][x] = GameFrame._GRID_OPEN

    def _estimate_cost(self, point):
        return self._grid.shape[0] - point[1]

    def _find_path(self):
        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        openset = [AStar_Node((0,0), None, 0, self._estimate_cost((0,0)))]
        visited = set()

        timeout = time.time()

        self._path = []
        while len(openset) > 0:
            if time.time() - timeout > GameFrame._maxPathTime:
                break
            curr = heapq.heappop(openset)
            visited.add(curr.point)
            if self._grid[curr.point[1],curr.point[0]] == GameFrame._GRID_OOB or curr.point[1] == self._grid.shape[0] - 1:
                while curr is not None:
                    self._path.append(curr.point)
                    curr = curr.prev
                self._path.reverse()
                return
            for dir in dirs:
                newpoint = ((curr.point[0] + dir[0]) % self._grid.shape[1], curr.point[1] + dir[1])
                if newpoint[1] < 0:
                    continue
                if newpoint in visited:
                    continue
                if self._grid[newpoint[1],newpoint[0]] == GameFrame._GRID_BLOCKED:
                    continue
                stepCost = 1 + 1.2 * curr.point[1]
                newnode = AStar_Node(newpoint, curr, curr.pathlen + 1, curr.pathlen + stepCost + self._estimate_cost(newpoint))
                heapq.heappush(openset, newnode)
                
    def showPlottedPath(self):
        rgbthresh = cv.cvtColor(self._thresh, cv.COLOR_GRAY2BGR)
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):
                if not self._grid[y][x] == GameFrame._GRID_OPEN:
                    continue

                realx, realy = self._real_coords[y, x]

                if x == 0 and y == 0:
                    cv.circle(rgbthresh, (realx, realy), 2, (0, 0, 255), -1)
                else:
                    cv.circle(rgbthresh, (realx, realy), 2, (255, 0, 255), -1)
                

        if len(self._path) > 0:
            for i in range(len(self._path) - 1):
                curr = self._path[i]
                next = self._path[i+1]

                realcurr = self._real_coords[curr[1], curr[0]]
                realnext = self._real_coords[next[1], next[0]]

                cv.arrowedLine(rgbthresh, realcurr, realnext, (0, 255, 0), 3)
        
        cv.drawContours(rgbthresh, [self._player_contour], 0, (0, 255, 255), -1)
        nextMove = self.getNextMove()
        if nextMove is not None:
            cv.putText(rgbthresh, nextMove, (self._dims[1]-345,61), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(rgbthresh, "Grid {}".format(self._gridPrecisionLevel), (10, rgbthresh.shape[0] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv.putText(rgbthresh, "Threshold {} {}".format(self._debugThreshInfo[0], self._debugThreshInfo[1]), (10, rgbthresh.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        return rgbthresh

    def getNextMove(self):
        if len(self._path) < 2:
            return None
        
        dx = self._path[1][0] - self._path[0][0]
        dy = self._path[1][1] - self._path[0][1]

        if abs(dx) > 1:
            if dx > 0:
                dx = -1
            else:
                dx = 1

        assert (dx == 0 or abs(dx) == 1) and (dy == 0 or abs(dy) == 1)
        assert dx != dy

        if dx == 0:
            if dy == 1:
                return "OUT"
            elif dy == -1:
                return "IN"
            else:
                raise Exception("Bad dy")
        elif dx == 1:
            return "LEFT"
        elif dx == -1:
            return "RIGHT"
        else:
            raise Exception("Bad dx")
