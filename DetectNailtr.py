import cv2
from cv2 import imshow
from matplotlib.pyplot import contour
import numpy as np

from random import randint
import matplotlib.pyplot as plt

from matplotlib.contour import ContourSet
from matplotlib.pyplot import contour

import numpy as np
import skimage.filters as sk_filters
from scipy import ndimage
import imutils
import time
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse




def DetectPositionMaxSkin(filename,xc, yc, wc, hc, lower, upper):
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(ap.parse_args())
    pts = deque(maxlen=args["buffer"])  

    #y=y+50
    Image = cv2.VideoCapture(filename)
    
    #Image = cv2.VideoCapture('t8.mp4')
    success, frame = Image.read()
    
    while success :
        success, frame = Image.read()
        #cv2.imshow('Imagem Original', frame)
        if success:
            
            cropeedIMAGE = frame[yc:yc+hc, xc:xc+wc]
            
            converted = cv2.cvtColor(cropeedIMAGE, cv2.COLOR_BGR2HSV)
            #cv2.imshow('convertedHSV',converted)
            skinMask = cv2.inRange(converted, lower, upper)
            #cv2.imshow('skin',skinMask)

            # apply a series of erosions and dilations to the mask
            # using an elliptical kernel            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            skinMask = cv2.erode(skinMask, kernel, iterations=1)
            skinMask = cv2.dilate(skinMask, kernel, iterations=1)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (13, 13), 5)
            #cv2.imshow('skinMask',skinMask)
            #skin = cv2.bitwise_and(cropeedIMAGE, cropeedIMAGE, mask=skinMask)
            #cv2.imshow('skin',skin)


            ########################################################

            #nails = cv2.bitwise_and(cropeedIMAGE, cropeedIMAGE, mask=skinMask)
            #cv2.imshow('nails',nails)

            #lowerFinger =np.array([8, 15, 110], dtype="uint8")
            #upperFinger = np.array([8, 15, 110], dtype="uint8")
                
            hsv_img = cv2.cvtColor(cropeedIMAGE, cv2.COLOR_BGR2HSV)
            #hsv_img = cv2.inRange(hsv_img, lowerFinger, upperFinger)
            #cv2.imshow('hsv_img', hsv_img)
            #mask = cv2.inRange(hsv_img,(10,15,110),(30,255,255))
            #cv2.imshow('mask', mask)

            # Extracting Saturation channel on which we will work
            img_s = hsv_img[:, :, 1]
            #img_s = skin[:, :, 1]
            #cv2.imshow('img_s', img_s)

            # smoothing before applying  threshold
            img_s_blur = cv2.GaussianBlur(img_s, (7,7), 0)  
            #img_s_blur = cv2.medianBlur(skin,5)
            #cv2.imshow('img_s_blur', img_s_blur)
            
            img_s_binary = cv2.threshold(img_s_blur,100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Thresholding to generate binary image (ROI detection)
            #cv2.imshow('img_s_binary1', img_s_binary)

            # reduce some noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            img_s_binary = cv2.morphologyEx(img_s_binary, cv2.MORPH_OPEN, kernel, iterations=4) 
            #cv2.imshow('img_s_binary1', img_s_binary)
            
            # ROI only image extraction & contrast enhancement, you can crop this region 
            img_croped = cv2.bitwise_and(img_s, img_s_binary) * 4  
            #cv2.imshow('img_croped', img_croped)
            
             #  eliminate
            kernel = np.ones((5, 5), np.float32)/25
            processedImage = cv2.filter2D(img_s_binary, -1, kernel)
            img_s_binary[processedImage > 250] = 0
            #cv2.imshow('img_s_binary2', img_s_binary)

            
            abs_grad_x = cv2.convertScaleAbs(cv2.Sobel(img_croped, cv2.CV_64F, 1, 0, ksize=3))
            abs_grad_y = cv2.convertScaleAbs(cv2.Sobel(img_croped, cv2.CV_64F, 0, 1, ksize=3))
            grad = cv2.addWeighted(abs_grad_x, .7, abs_grad_y, .7, 0)  # Gradient calculation
            grad = cv2.medianBlur(grad, 13)



            edges = cv2.threshold(grad, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            cv2.imshow('edges',edges)


            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
            #skinMask2 = cv2.erode(edges, kernel, iterations=3)
            #skinMask2 = cv2.dilate(edges, kernel, iterations=3)
            #cv2.imshow('skinMask2',skinMask2)

            cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            cnts = imutils.grab_contours(cnts)
	
            center = None
            if len(cnts) > 0:
				# find the largest contour in the mask, then use
				# it to compute the minimum enclosing circle and
				# centroid
                c = max(cnts, key=cv2.contourArea)

                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				# only proceed if the radius meets a minimum size
                if radius > 1:
                    #xb, yb, wb, hb= cv2.boundingRect(c);
                    #cv2.rectangle(cropeedIMAGE, (xb, yb), (xb + wb, yb + hb), (255, 0, 0), 2)  # blue
                   
                    min_rect = cv2.minAreaRect(c)  # min_area_rectangle
                    min_rect = np.int0(cv2.boxPoints(min_rect))
                    cv2.drawContours(cropeedIMAGE, [min_rect], 0, (0, 255, 0), 2)  # green
					
                    # draw the circle and centroid on the frame,
					# then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                    cv2.circle(cropeedIMAGE, center, 5, (0, 0, 255), -1)
			# update the points queue
            pts.appendleft(center)
				# loop over the set of tracked points
            for i in range(1, len(pts)):
				# if either of the tracked points are None, ignore
				# them
                if pts[i - 1] is None or pts[i] is None:
                    continue
				# otherwise, compute the thickness of the line and
				# draw the connecting lines
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                #cv2.line(cropeedIMAGE, pts[i - 1], pts[i], (0, 0, 255), thickness)

                #cv2.circle(frame, (int(thickness), int(thickness)), int(radius), (0, 0, 255), 2)

            cv2.imshow("Frame", cropeedIMAGE)

    

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
#xc,yc,wc,hc,skin,skinMask,hsv_img,img_s_blur,img_s_binary1,img_croped,edges,cropeedIMAGE