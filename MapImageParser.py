import numpy as np
import cv2
from HoughBundler import HoughBundler as HB

MIN_LEN = 35

class NoIntersectionError(Exception):
    pass

class InfiniteResultOfIntersectionError(Exception):
    pass

def delete_dots(lines, thresh):
    ls = lines.copy()
    ls = sorted(ls, key=lambda l: distance(*l[:2], *l[2:]))
    for i in range(len(ls)):
        if distance(*ls[i][:2], *ls[i][2:]) > thresh:
            del ls[0:i]
            break
    return ls

def force_aproximate(x1,y1,x2,y2):
    if 5 < np.degrees(np.arctan2(abs(y1-y2), abs(x1-x2))) < 85:
        return (x1, y1, x2, y2)
    if abs(x1 - x2) > abs(y1 - y2):
        return (x1, min(y1, y2), x2, min(y1, y2))
    else:
        return (min(x1, x2), y1, min(x1, x2), y2)

def force_align(line1, line2, threshold):
    if line1[0] == line1[2]:
        if abs(line1[0] - line2[0]) < threshold:
            return (line1, (line1[0], line2[1], line1[2], line2[3]))
        else:
            return (line1, line2)
    if line1[1] == line1[3]:
        if abs(line1[1] - line2[1]) < threshold:
            return (line1, (line2[0], line1[1], line2[2], line1[3]))
        else:
            return (line1, line2)
    return (line1, line2)

def distance(x1, y1, x2, y2):
    return ( (x1 - x2) ** 2 + (y1 - y2) ** 2 ) ** 0.5

def line_distance(l, p):
    a = distance(*l)
    b = distance(*l[:2], *p)
    c = distance(*l[2:], *p)
    p = (a + b + c)/2
    s = (p * (p - a) * (p - b) * (p - c)) ** 0.5
    return 2 * s / a

def intersection_point(phi1, h1, phi2, h2):
    if abs(phi1 - phi2) < 0.01:
        if abs(h2 - h1) < 0.01:
            raise InfiniteResultOfIntersectionError("The same line given")
        raise NoIntersectionError("Lines have no intercetion point")
    x = (h2 - h1) / (np.tan(phi1) - np.tan(phi2))
    y = x * np.tan(phi1) + h1
    return (int(x),int(y))

def line_parametres(x1,y1,x2,y2):
    phi = np.arctan2(y2 - y1, x2 - x1)
    h   = y1 - x1 * np.tan(phi)
    return (phi, h)

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

erode_kernel  = get_kernel(3)
dilate_kernel = get_kernel(1)
hough_bundler = HB()

#static shit happens
class MapImageParser:

#    def edge_boarders(cls, boarders, cx, cy):
#        
#        h_e = sorted(boarders, key=lambda l: line_distance(l))
    
    @classmethod
    def aproximate_boarders(cls, m_lines):
        apro_lines = []
        
        for i in range(len(m_lines)):
            p1, p2 = m_lines[i]
            apro_lines.append(force_aproximate(*p1, *p2))
        
        for i in range(len(apro_lines)):
            for j in range(i+1, len(apro_lines)):
                line1, line2 = apro_lines[i], apro_lines[j]
                if ((line1[1] == line1[3]) and (line2[1] == line2[3])) or \
                    (line1[0] == line1[2]) and (line2[0] == line2[2]):
                        apro_lines[i], apro_lines[j] = force_align(line1, line2, 20)
        return apro_lines
    
    @classmethod
    def get_boarders(cls, img, threshold):
        lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 4, 
                            threshold=40, minLineLength=MIN_LEN, maxLineGap=40)
        
        m_lines = hough_bundler.process_lines(lines, threshold)
        
        apro_lines = []
        
        for i in range(len(m_lines)):
            p1, p2 = m_lines[i]
            apro_lines.append(force_aproximate(*p1, *p2))
        
        for i in range(len(apro_lines)):
            for j in range(i+1, len(apro_lines)):
                line1, line2 = apro_lines[i], apro_lines[j]
                if ((line1[1] == line1[3]) and (line2[1] == line2[3])) or \
                    (line1[0] == line1[2]) and (line2[0] == line2[2]):
                        apro_lines[i], apro_lines[j] = force_align(line1, line2, 20)
        return apro_lines
    
    @classmethod
    def parse(cls, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        board_thresh  = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 0),   (255, 255, 214)))
        markup_thresh = cv2.bitwise_not(cv2.inRange(hsv, (0, 5, 137), (255, 253, 255)))
        thresh = cv2.bitwise_and(board_thresh, markup_thresh)
    
        morph = cv2.erode(thresh, kernel=erode_kernel,  iterations=1)
        morph = cv2.dilate(morph, kernel=dilate_kernel, iterations=1)
        
        morph = cv2.resize(morph,    (640, 640))
        img   = cv2.resize(img, (640, 640))
        
        lines = cv2.HoughLinesP(morph, 1, np.pi / 180, 50, 300, 100, 10)
        
        h_line = sorted(lines, key=lambda line: max(line.ravel()[1], line.ravel()[3]))[-1].ravel()
        phi = np.arctan2(abs(h_line[3] - h_line[1]),abs(h_line[2] - h_line[0]))*180/np.pi
        
        morph = rotate_image(morph, phi)
        img   = rotate_image(img,   phi)
        
        lines = cv2.HoughLinesP(morph, rho=1, theta=np.pi / 180, threshold=80, 
                                minLineLength=100, maxLineGap=10)
    
        h_lines = sorted(lines, key=lambda line: max(line.ravel()[1], line.ravel()[3]))
        d_line  = h_lines[-1].ravel()
        u_line  = h_lines[0].ravel()
        
        x_lines = sorted(lines, key=lambda l: max(l.ravel()[0], l.ravel()[2]))
        r_line  = x_lines[-1].ravel()
        l_line  = x_lines[0].ravel()
    
        p1 = intersection_point(*line_parametres(*r_line), *line_parametres(*d_line))
        p2 = intersection_point(*line_parametres(*l_line), *line_parametres(*u_line))
        p3 = intersection_point(*line_parametres(*r_line), *line_parametres(*u_line))
        p4 = intersection_point(*line_parametres(*l_line), *line_parametres(*d_line))
        
        w,h = img.shape[:2]
        m   = 25
        
        src_sq = np.array((p1, p2, p3, p4), dtype=np.float32)
        dst_sq = np.array(((w-m,h-m), (0, 0), (w-m, 0), (0, h-m)), dtype=np.float32)
    
        nohomo, _ = cv2.findHomography(src_sq, dst_sq)
        morph = cv2.warpPerspective(morph, nohomo, (640, 640))
        img   = cv2.warpPerspective(img, nohomo, (640, 640))
        return img, cv2.erode(morph, kernel=dilate_kernel, iterations=1)

image        = cv2.imread("maps/map2.jpg")
image, morph = MapImageParser.parse(image)
boarders     = MapImageParser.get_boarders(image, morph)
edge_boarders = sorted(boarders, key=lambda l: line_distance(
        l, (image.shape[0]/2, image.shape[1]/2)), reverse=True)[:8]

#boarders = sorted(boarders, key=lambda l: distance(*l[0], *l[1]))

#horizontals = list(filter(lambda l: l[0] == l[2], boarders))
#verticals   = list(filter(lambda l: l[1] == l[3], boarders))

boarders = delete_dots(boarders, MIN_LEN)

for line in boarders:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1,y1), (x2, y2), (255, 0, 0), 3)


for line in edge_boarders:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1,y1), (x2, y2), (0, 0, 255), 3)

cv2.imshow("map2", image)
cv2.imshow("map1 threshold", morph)
cv2.waitKey(0)
cv2.destroyAllWindows()