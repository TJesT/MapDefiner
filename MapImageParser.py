import numpy as np
import cv2
from HoughBundler import HoughBundler as HB

MIN_LEN = 35

class IntersectionError(Exception):
    pass

def contours(morph):
    contours, _ = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#    print(len(contours))
    cntrs = []
    l_c = (320,320)
    for c in sorted(contours, key=lambda x: distance(0,0,
        (min(x.ravel()[0::2])+max(x.ravel()[0::2]))/2, 
        (min(x.ravel()[1::2])+max(x.ravel()[1::2]))/2)):
        max_x = max(c.ravel()[0::2])
        min_x = min(c.ravel()[0::2])
        max_y = max(c.ravel()[1::2])
        min_y = min(c.ravel()[1::2])
        n_l_c   = ((max_x + min_x)/2, (max_y + min_y)/2)
        if distance(*l_c, *n_l_c) < 75 or \
        distance(min_x, min_y, max_x, max_y) < MIN_LEN * 2 ** 0.5 or \
        distance(320, 320, *n_l_c) < 30:
#            print("l_c:", l_c, "n_l_c:", n_l_c, "point:", (min_x, min_y, max_x, max_y))
            continue
        else:
#            print("l_c:", l_c, "n_l_c:", n_l_c, "point:", (min_x, min_y, max_x, max_y))
            cntrs.append((min_x, min_y, max_x, max_y))
            l_c = n_l_c
    return cntrs

def boarders_from_contours(c):
    b = []
    for c_ in c:
        x1, y1, x2, y2 = c_
        b += [(x1,y1,x1,y2), 
              (x1,y2,x2,y2), 
              (x2,y2,x2,y1), 
              (x2,y1,x1,y1)]
    return b

def delete_dots(lines, thresh):
    ls = lines.copy()
    ls = sorted(ls, key=lambda l: distance(*l[:2], *l[2:]))
    for i in range(len(ls)):
        if distance(*ls[i][:2], *ls[i][2:]) > thresh:
            del ls[0:i]
            break
    return ls

def nearest_intersections(line, a_lines):
    all_intersections = []
    phi, h = line_parametres(*line)
    cx, cy = (line[0] + line[2])/2, (line[1] + line[3])/2
    for l in a_lines:
        try:
            phi2, h2 = line_parametres(*l)
            phi2 = (phi2 if abs(h2) < 100000 else l[0])
            all_intersections.append(intersection_point(
                    (phi if abs(h) < 100000 else line[0]), h, phi2, h2))
#            print("fucking intersectors", *l)
        except IntersectionError:
            continue
    all_intersections = list(set(all_intersections))
    all_intersections = sorted(all_intersections, key=lambda p: distance(cx, cy, *p))
    
    p2 = all_intersections[1]
    p1 = all_intersections[0]
    n_line = p1 + (p2 if distance(*p1, *p2) > 10 else all_intersections[2])
    return n_line

def board_rotate(boards, cx, cy, reverse=False):
    b = boards.copy()
    r = -1 if reverse else 1
    for i in range(len(b)):
        n = get_normal(b[i], 5)
        x1, y1, x2, y2 = b[i]
        lx, ly = int((x1 + x2)/2), int((y1 + y2)/2)
        if r*distance(cx, cy, lx + n[0], ly + n[1]) > r*distance(cx, cy, lx, ly):
            b[i] = (b[i][2], b[i][3], b[i][0], b[i][1])
    return b

def get_normal(l, length=1):
    x1, y1, x2, y2 = l
    a = y2 - y1
    b = x2 - x1
    p = np.degrees(np.arctan2(a, b)) + 90 
    return (np.cos(np.radians(p))*length, np.sin(np.radians(p))*length)

def force_aproximate(x1,y1,x2,y2):
    if 10 < np.degrees(np.arctan2(abs(y1-y2), abs(x1-x2))) < 80:
        return (x1, y1, x2, y2)
    if abs(x1 - x2) > abs(y1 - y2):
        return (x1, min(y1, y2), x2, min(y1, y2))
    else:
        return (min(x1, x2), y1, min(x1, x2), y2)

def force_align(line1, line2, threshold=0):
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
    if abs(np.degrees(phi1 - phi2)) < 1:
        if abs(h2 - h1) < 1:
            raise IntersectionError("The same line given")
        raise IntersectionError("Lines have no intercetion point")
    if abs(h1) > 100000:
#        print("1")
        return (phi1, int(np.tan(phi2)*phi1 + h2))
    if abs(h2) > 100000:
#        print("2")
        return (phi2, int(np.tan(phi1)*phi2 + h1))
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
 
    return cv2.warpAffine(image, M, (640, 640))

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
                    ((line1[0] == line1[2]) and (line2[0] == line2[2])):
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
        return img, morph#cv2.erode(morph, kernel=dilate_kernel, iterations=1)

image        = cv2.imread("maps/7.jpg")
image, morph = MapImageParser.parse(image)
#cv2.imshow("1", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows

boarders     = MapImageParser.get_boarders(image, morph)
boarders     = sorted(boarders, key=lambda l: line_distance(
        l, (image.shape[0]/2, image.shape[1]/2)), reverse=True)
edge_boarders, boarders = boarders[:8], boarders[8:]
#edge_boarders = board_rotate(edge_boarders, image.shape[0]/2, image.shape[1]/2)

e_b_1 = edge_boarders[:4]
e_b_2 = edge_boarders[4:]

#boarders = sorted(boarders, key=lambda l: distance(*l[0], *l[1]))

#horizontals = list(filter(lambda l: l[0] == l[2], boarders))
#verticals   = list(filter(lambda l: l[1] == l[3], boarders))

#boarders = delete_dots(boarders, MIN_LEN)

boarders = boarders_from_contours(contours(morph))

n_b = [nearest_intersections(i, e_b_2) for i in e_b_1]+[nearest_intersections(i, e_b_1) for i in e_b_2]
n_b = board_rotate(n_b, image.shape[0]/2, image.shape[1]/2)
#a = nearest_intersections(edge_boarders[0], edge_boarders[1:])
#b = nearest_intersections(e_b_1[1], e_b_2)
#
#cv2.line(image, a[:2], a[2:], (255,255,0), 3)
#cv2.line(image, b[:2], b[2:], (255,255,0), 3)

for line in boarders: # boarders:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1,y1), (x2, y2), (255, 0, 0), 3)
    xc, yc = int((x1 + x2)/2), int((y1 + y2)/2)
    n = get_normal(line, 25)
    cv2.line(image, (xc, yc), (xc + int(n[0]), yc + int(n[1])), (0, 255, 255), 3)

for line in n_b:
    x1, y1, x2, y2 = line
    xc, yc = int((x1 + x2)/2), int((y1 + y2)/2)
    n = get_normal(line, 25)
    cv2.line(image, (x1,y1), (x2, y2), (0, 0, 255), 3)
    cv2.line(image, (xc, yc), (xc + int(n[0]), yc + int(n[1])), (255, 0, 255), 3)

cv2.imshow(str(image.shape[:2]), image)
cv2.imshow("map1 threshold",     morph)
cv2.waitKey(0)
cv2.destroyAllWindows()