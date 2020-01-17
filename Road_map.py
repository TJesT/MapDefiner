from MapImageParser import MapImageParser

class Roadmap():
    def  __init__(self, read_img, x, y):
        parse_map_img(read_img) # построить граф
        self.robot_x = x
        self.robot_y = y
        return
    
    def getTangent(self):
        return 0.0
	
    def setTrace(self, time, alpha, last_v, v):
        return
        #дай мне новую направляюую
        #!! не я делаю !!
	
    def getTrace():
        #!! не я делаю !!
        pass
