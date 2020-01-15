import numpy as np
import cv2
import time
from win32api import GetSystemMetrics

WINDOW_SIZE = 640

w       = GetSystemMetrics(0)
h       = GetSystemMetrics(1)
kernels = []
try:
    open("kernels.txt").close()
except FileNotFoundError:
    open("kernels.txt", "w").close()
with open("kernels.txt") as op:
    for line in op.readlines():
        l = list(map(np.uint8, line.split()))
        k_size = int(len(l) ** 0.5)
        kernels.append(np.array([[ l[i*k_size + j] for j in range(k_size)] 
                        for i in range(k_size)], dtype=np.uint8)) 

def recollect_kernels():
    global kernels
    kernels = []
    with open("kernels.txt") as op:
        for line in op.readlines():
            l = list(map(np.uint8, line.split()))
            k_size = int(len(l) ** 0.5)
            kernels.append(np.array([[ l[i*k_size + j] for j in range(k_size)] 
                        for i in range(k_size)], dtype=np.uint8)) 


def thresh_bars_hsv(window_name, hsv_image):
    global w, h
    sn = "HSV settings"
    
    cv2.namedWindow(window_name)
    cv2.namedWindow(sn)
    hsv_img = cv2.resize(hsv_image, (w, h))
    cv2.createTrackbar("Hl", sn, 0, 255, (lambda a: None))
    cv2.createTrackbar("Hu", sn, 0, 255, (lambda a: None))
    cv2.createTrackbar("Sl", sn, 0, 255, (lambda a: None))
    cv2.createTrackbar("Su", sn, 0, 255, (lambda a: None))
    cv2.createTrackbar("Vl", sn, 0, 255, (lambda a: None))
    cv2.createTrackbar("Vu", sn, 0, 255, (lambda a: None))
    while True:
        Hl, Hu = cv2.getTrackbarPos("Hl", sn), cv2.getTrackbarPos("Hu", sn)
        Sl, Su = cv2.getTrackbarPos("Sl", sn), cv2.getTrackbarPos("Su", sn)
        Vl, Vu = cv2.getTrackbarPos("Vl", sn), cv2.getTrackbarPos("Vu", sn)

        threshold = cv2.inRange(hsv_img, (Hl, Sl, Vl), (Hu, Su, Vu))

        cv2.imshow(window_name, threshold)

        if cv2.waitKey(1) & 0xFF == 13:
            break

        time.sleep(0.005)
        
def erode_dilate_bars(window_name, thresh_image):
    recollect_kernels()
    
    global w, h, kernels
    
    edn   = "Erode/Dilate settings"
    k_len = len(kernels)
    
    cv2.namedWindow(window_name)
    cv2.namedWindow(edn)
    thresh_image = cv2.resize(thresh_image, (w, h))
    
    cv2.createTrackbar("Ei", edn, 0, 10, (lambda a: None))
    cv2.createTrackbar("Ek", edn, 0, k_len - 1, (lambda a: None))
    cv2.createTrackbar("Di", edn, 0, 10, (lambda a: None))
    cv2.createTrackbar("Dk", edn, 0, k_len - 1, (lambda a: None))
    
    while True:
        Ei, Ek = cv2.getTrackbarPos("Ei", edn), kernels[cv2.getTrackbarPos("Ek", edn)]
        Di, Dk = cv2.getTrackbarPos("Di", edn), kernels[cv2.getTrackbarPos("Dk", edn)]
                
        morph = cv2.erode(thresh_image, kernel=Ek, iterations=Ei)
        morph = cv2.dilate(morph, kernel=Dk, iterations=Di)
        
        cv2.imshow(window_name, morph)

        if cv2.waitKey(1) & 0xFF == 13:
            break

        time.sleep(0.005)

def kernel_bars(window_name, thresh_image, kernel_size):
    kernel = np.zeros([kernel_size, kernel_size])
    mrphn = "Morphology Settings"
    
    def on_mouse_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            kernel[y % kernel_size, x % kernel_size] += 1
            kernel[y % kernel_size, x % kernel_size] %= 2
    
    cv2.namedWindow("kernel", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("kernel", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("kernel", WINDOW_SIZE, WINDOW_SIZE)
    
    cv2.setMouseCallback("kernel", on_mouse_callback)
    
    while True:
        cv2.imshow("kernel", kernel)

        if cv2.waitKey(1) & 0xFF == 13:
            break
    
    cv2.destroyAllWindows()
    
    cv2.namedWindow(mrphn, cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow(window_name, WINDOW_SIZE, WINDOW_SIZE)
    
    cv2.createTrackbar("Ei", mrphn, 0, 10, (lambda a: None))
    cv2.createTrackbar("Di", mrphn, 0, 10, (lambda a: None))
    
    k = " ".join(list(map(str, map(int, kernel.ravel()))))
    write = False
    
    with open("kernels.txt", "r") as op:
        if k + "\n" not in op.readlines():
            write = True
        op.close()
    
    if write:
        with open("kernels.txt", "a") as wr:
            wr.write(k + "\n")
            wr.close()
    
    kernel = np.array(kernel, dtype=np.uint8)
    
    while True:
        Ei, Di = cv2.getTrackbarPos("Ei",  mrphn), cv2.getTrackbarPos("Di",  mrphn)
        
        morph = cv2.erode(thresh_image, kernel=kernel, iterations=Ei)
        morph = cv2.dilate(morph, kernel=kernel, iterations=Di)
        
        cv2.imshow(window_name, morph)
        
        if cv2.waitKey(1) & 0xFF == 13:
            break

        time.sleep(0.005)
        
    cv2.destroyAllWindows()

#kernel_bars("vgolovu", cv2.imread("../letters/j.png"), int(input()))
