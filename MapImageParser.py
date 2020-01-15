#import numpy as np
#import cv2
from controlBars import thresh_bars_hsv as tbh

#static shit happens
class MapImageParser:
    img   = None
    graph = None
    
    @classmethod
    def parse(cls):
        tbh("1", cls.img)
        return
