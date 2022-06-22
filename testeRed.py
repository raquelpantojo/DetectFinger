

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



def DetectPositionMaxSkin(filename,x, y, w, h): 
	
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help="path to the (optional) video file")
	ap.add_argument("-b", "--buffer", type=int, default=64,
		help="max buffer size")
	args = vars(ap.parse_args())


	pts = deque(maxlen=args["buffer"])  
	
	Image = cv2.VideoCapture(filename)
	success, frame = Image.read()

	while success :
		success, frame = Image.read()
		if success:
			#cropeedIMAGE = frame[y:y+h, x:x+w]
			#cv2.imshow('hsv',cropeedIMAGE)	
			#blurred = cv2.GaussianBlur(cropeedIMAGE, (11, 11), 0)
			#cv2.imshow('hsv',blurred)	
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			#cv2.imshow('hsv',hsv)
		
			mask = cv2.inRange(hsv,(10,15,110),(30,255,255))
			#cv2.imshow('hsvdilate',mask)

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
			skinMask = cv2.erode(mask, kernel, iterations=3)
			skinMask = cv2.dilate(mask, kernel, iterations=3)

			cv2.imshow('hsvdilate',skinMask)

			cnts = cv2.findContours(skinMask.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
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
				if radius > 2:
					# draw the circle and centroid on the frame,
					# then update the list of tracked points
					cv2.circle(frame, (int(x), int(y)), int(radius),
						(0, 255, 255), 2)
					cv2.circle(frame, center, 5, (0, 0, 255), -1)
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
				cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

			cv2.imshow("Frame", frame)
    

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
cv2.destroyAllWindows()