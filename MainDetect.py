import DetectSkin
import DetectNailtr
import DetectNailtrv2
import DetectNail
import DetectNailRet
import testeRed
from matplotlib.contour import ContourSet
from matplotlib.pyplot import contour

import numpy as np
import skimage.filters as sk_filters
from scipy import ndimage

# define the upper and lower boundaries of the HSV pixel
# intensities to be consired 'skin'
lower = np.array([8, 15, 150], dtype="uint8")
upper = np.array([30, 255, 255], dtype="uint8")


#filename = 't8.mp4'
filename = 'v1102.mp4'
#filename = 'v3.mp4'
#filename = 'VRPS.mp4'
#filename = 'testtampinha.mp4'
#filename = 'blackskin.wmv'


frame_number = 30

# Cria um retangulo que seleciona somente a max area da pele:
x , y, w, h = DetectSkin.DetectPositionMaxSkin(filename, frame_number, lower, upper)
#xc=x+50    
#yc=y+50
#wc=w-100
#hc=h-100   
# detecta a região da ponta do dedo:
#testeRed.DetectPositionMaxSkin(filename,x, y, w, h)
#DetectNail.DetectPositionMaxSkin(filename,xc, yc, wc, hc,lower, upper)
#DetectNail.DetectPositionMaxSkin(filename,x, y, w, h,lower, upper)

# Cria um retngulo na ponta do dedo
lowerhsv = np.array([200, 0, 127], dtype="uint8")
upperhsv = np.array([255, 10, 130], dtype="uint8")

DetectNailRet.DetectPositionMaxSkin(filename,x, y, w, h,lowerhsv, upperhsv)

#DetectNailtrv2.DetectPositionMaxSkin(filename,x, y, w, h,lower, upper)
#cropeedIMAGE=DetectSkin.croppedSkin(filename,xc,yc,wc,hc)


#final_frame = cv2.hconcat((skin, skinMask))
#Show the concatenated frame using imshow.
#cv2.imshow('frame',skin)
