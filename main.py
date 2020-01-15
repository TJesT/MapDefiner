from os.path import isfile, join
from os import listdir
from controlBars import thresh_bars_hsv as tbh, erode_dilate_bars as edb
#import MapImageParser as MIP
import numpy as np
import cv2

def rotate_image(image, angle):
    h, w = image.shape[:2]
    cX, cY = w//2, h//2
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH))

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
    #img = cv2.resize(data[fn], (640, 640))
    
    lines = cv2.HoughLinesP(morph, 1, np.pi / 180, 50, 300, 100, 0)
    
    h_line = sorted(lines, key=lambda line: max(line.ravel()[1], line.ravel()[3]))[-1].ravel()
    phi = np.arctan2(abs(h_line[3] - h_line[1]),abs(h_line[2] - h_line[0]))*180/np.pi
    print(phi)
    
    morph = rotate_image(morph, phi)
    
#    cv2.line(img,(h_line[0], h_line[1]),(h_line[2],h_line[3]),(0,255,0),2)
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow(fn + " thresholded", morph)

cv2.waitKey(0)

cv2.destroyAllWindows()
#MIP.img = hsv_data["map1.jpg"]
#MIP.parse()