from shapely.geometry import Point, LineString, Polygon
import time


class Timer:

    def start(self):
        self.start = time.perf_counter()

    def printTime(self, msg):
        self.end = time.perf_counter()
        print("[" + msg + "] Using Time: ", self.end - self.start)
        print("------------------------------------\n")


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

    def log(self, title="Root"):
        print("\n[" + title + "] Sub Wirelength: ", self.sub_wl)
        print("[" + title + "] Total: ", self.total_wl)