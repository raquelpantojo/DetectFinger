
import cv2
from matplotlib.pyplot import contour
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.pyplot as plt

#Teste Github

def DetectPositionMaxSkin(filename,x, y, w, h, lower, upper):
    #y=y+50
    Image = cv2.VideoCapture(filename)
    
    #Image = cv2.VideoCapture('t8.mp4')
    success, frame = Image.read()
    
    while success :
        success, frame = Image.read()
        #cv2.imshow('Imagem Original', frame)
        if success:
            cropeedIMAGE = frame[y:y+h, x:x+w]
            
            hsv_img = cv2.cvtColor(cropeedIMAGE, cv2.COLOR_BGR2HSV)
            #cv2.imshow('converted',converted)
            #skinFinger = cv2.inRange(converted, lower, upper)

            img_s = hsv_img[:, :, 1]  # Extracting Saturation channel on which we will work
            #cv2.imshow('img_s',img_s)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            img_s_blur = cv2.GaussianBlur(img_s, (7, 7), 0)
            #cv2.imshow('skinMask',skinMask) 
            img_s_binary = cv2.threshold(img_s_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Thresholding to generate binary image (ROI detection)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img_s_binary = cv2.morphologyEx(img_s_binary, cv2.MORPH_OPEN, kernel, iterations=3)  # reduce some noise
            

            img_croped = cv2.bitwise_and(img_s, img_s_binary) * 2  # ROI only image extraction & contrast enhancement, you can crop this region 
            abs_grad_x = cv2.convertScaleAbs(cv2.Sobel(img_croped, cv2.CV_64F, 1, 0, ksize=3))
            abs_grad_y = cv2.convertScaleAbs(cv2.Sobel(img_croped, cv2.CV_64F, 0, 1, ksize=3))
            grad = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, .5, 0)  # Gradient calculation
            grad = cv2.medianBlur(grad, 13)
            cv2.imshow('grad',grad)



            edges = cv2.threshold(grad, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cv2.imshow('edges',edges)

            cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contours Detection
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnt = None
            max_area = 0
            for c in cnts:
                area = cv2.contourArea(c)
                if area > max_area:  # Filtering contour
                    max_area = area
                    cnt = c

            cv2.drawContours(hsv_img, [cnt], 0, (0, 255, 0), 3)
            cv2.imshow('hsv_img',hsv_img)
            
    


        
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    #xc,yc,wc,hc,skin,skinMask,hsv_img,img_s_blur,img_s_binary1,img_croped,edges,cropeedIMAGE
