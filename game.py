import cv2 as cv
import numpy as np
import math
import heapq

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

class GameFrame():
    too_close = 15
    dr = 20
    dtheta = 10

    _GRID_OPEN = 4
    _GRID_OOB = 2
    _GRID_BLOCKED = 1

    def __init__(self, frame):
        if frame is None:
            raise ValueError("No frame provided")
        self._frame = frame
        self._dims = frame.shape
        self._center = (self._dims[1] / 2, self._dims[0] / 2)

        self._path = []

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
        self._real_coords = np.zeros( (self._grid.shape[0], self._grid.shape[1], 2), np.int32)

        player_dist = float('inf')
        player_closest = (-1, -1)
        
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):            
                r = (GameFrame.dr * y) + self._center_radius
                theta = GameFrame.dtheta * x
                
                realx = int(self._center[0] + r * math.cos(math.radians(theta)))
                realy = int(self._center[1] - r * math.sin(math.radians(theta)))

                self._real_coords[y, x] = (realx, realy)

                if y == 0:
                    currdist = _point_dist((realx, realy), self._closest)
                    if currdist < player_dist:
                        player_dist = currdist
                        player_closest = (x, y)
                
                if realx < 0 or realx >= self._dims[1]:
                    self._grid[y][x] = GameFrame._GRID_OOB
                    continue
                if realy < 0 or realy >= self._dims[0]:
                    self._grid[y][x] = GameFrame._GRID_OOB
                    continue

                if y > 0:
                    area_around = self._thresh[max(0, realy-GameFrame.too_close):min(self._dims[0], realy+GameFrame.too_close),max(0, realx-GameFrame.too_close):min(self._dims[1], realx+GameFrame.too_close)]
                    if area_around.max() > 0:
                        self._grid[y][x] = GameFrame._GRID_BLOCKED
                        continue
                else:
                    if self._thresh[realy,realx] > 0:
                        self._grid[y][x] = GameFrame._GRID_BLOCKED
                        continue
                self._grid[y][x] = GameFrame._GRID_OPEN
        
        self._player_closest = player_closest

    def _estimate_cost(self, point):
        return self._grid.shape[0] - point[1]

    def findPath(self):
        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        closest = tuple(self._player_closest)
        openset = [AStar_Node(closest, None, 0, self._estimate_cost(closest))]
        visited = set()

        self._path = []
        while len(openset) > 0:
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
                newnode = AStar_Node(newpoint, curr, curr.pathlen + 1, curr.pathlen + 1 + self._estimate_cost(newpoint))
                heapq.heappush(openset, newnode)

                
    def showPlottedPath(self):
        rgbthresh = cv.cvtColor(self._thresh, cv.COLOR_GRAY2BGR)
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):
                if not self._grid[y][x] == GameFrame._GRID_OPEN:
                    continue

                realx, realy = self._real_coords[y, x]
                
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