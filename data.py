from shapely.geometry import Point, LineString, Polygon
from numpy import asarray


class DMENode:

    def __init__(self,
                 ms=None,
                 left=None,
                 right=None,
                 sub_wl=0.0,
                 total_wl=0.0):
        if ms is None or isinstance(ms, LineString) or isinstance(ms, Point):
            self.ms = ms
        else:
            raise TypeError('Merge segment class must be Point or LineString')
        self.left = left
        self.right = right
        self.father = None
        self.sub_wl = sub_wl
        self.total_wl = total_wl
        self.name = ""
        self.id = 0
