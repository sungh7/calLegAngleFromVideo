import cv2, math
import numpy as np
from PIL import Image


def reverseImage(im, predPoints):
    predPoints[:, 0] = 224 - predPoints[:, 0]
    im = np.array(im)[:, ::-1, :]
    return im, predPoints 

def drawPredPoints(im, predPoints):
    for point in predPoints:
        im = cv2.circle(np.array(im), 
                        tuple(point),
                        radius=2,
                        color=(255,0,0),
                        thickness=-1
                       )
    return im

def preprocessImage(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    im = im.resize((224,224))
    return np.array(im)

def cal_angle(knee_point=0, ground_point=0, ankle_point=0, 
              scaling=True, origin_width=0, origin_height=0):
    knee_x, _ = knee_point
    ground_x, ground_y = ground_point
    _, ankle_y = ankle_point

    x_ = abs(int(ground_x) - int(knee_x))
    y_ = abs(int(ankle_y) - int(ground_y))
    
    if scaling and to_width != 0 and to_height !=0:
        x_ = x_*to_width/224
        y_ = y_*to_height/224
    
    slope_ = np.sqrt(x_**2+y_**2)
    
    cos = x_ / slope_
    rad = np.arccos(cos)
    angle = math.degrees(rad)
    return angle
    