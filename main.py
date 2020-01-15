from os.path import isfile, join
from os import listdir
from controlBars import thresh_bars_hsv as tbh, erode_dilate_bars as edb
#import MapImageParser as MIP
import numpy as np
import cv2

def get_kernel(id_int):
    ret = []
    with open("kernels.txt") as op:
        line = op.readlines()[id_int]
        l = list(map(np.uint8, line.split()))
        k_size = int(len(l) ** 0.5)
        ret = np.array([[ l[i*k_size + j] for j in range(k_size)] 
            for i in range(k_size)], dtype=np.uint8)
        op.close()
    return ret

maps   = "maps/"
output = "out/"

erode_kernel  = get_kernel(3)
dilate_kernel = get_kernel(1)

#print(erode_kernel, dilate_kernel)

data = {
    fn: cv2.imread(join(maps, fn)) for fn in listdir(maps) if isfile(join(maps,fn))
}

hsv_data = {
    fn: cv2.cvtColor(data[fn], cv2.COLOR_BGR2HSV) for fn in data.keys()
}

#gray = cv2.cvtColor(hsv_data['map1.jpg'], cv2.COLOR_BGR2GRAY)

for fn in hsv_data.keys():
    board_thresh  = cv2.bitwise_not(cv2.inRange(hsv_data[fn], (0, 0, 0),   (255, 255, 214)))
    markup_thresh = cv2.bitwise_not(cv2.inRange(hsv_data[fn], (0, 5, 137), (255, 253, 255)))
    thresh = cv2.bitwise_and(board_thresh, markup_thresh)
#tbh("1", hsv_data['map1.jpg'])
#edb("1", thresh)
    morph = cv2.erode(thresh, kernel=erode_kernel,  iterations=1)
    morph = cv2.dilate(morph, kernel=dilate_kernel, iterations=1)

    morph = cv2.resize(morph, (640, 640))

    cv2.imshow(fn + " thresholded", morph)

cv2.waitKey(0)

cv2.destroyAllWindows()
#MIP.img = hsv_data["map1.jpg"]
#MIP.parse()