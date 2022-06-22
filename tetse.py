import numpy as np
import cv2
from sklearn.cluster import KMeans

from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt

## Section Two.1 : Function to Extract Skin Color

def extractSkin(filename):
    Image = cv2.VideoCapture(filename)
    frame_number=40
    # Defining HSV Threadholds
    lower_threshold = np.array([8, 15, 110], dtype="uint8")
    upper_threshold = np.array([30, 255, 255], dtype="uint8")
  
    # keep looping over the frames in the video
    # Get the total number of frames in the video.
    fps = Image.get(cv2.CAP_PROP_FPS)
    frame_count = Image.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = Image.read()

    while success and frame_number <= frame_count:

        if success:
            frame_number += fps
            Image.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # resize the frame, convert it to the HSV color space,
            # and determine the HSV pixel intensities that fall into
            # the speicifed upper and lower boundaries
            # frame = imutils.resize(frame, width=400)  # 400
            #frame=cv2.resize(frame, (640, 480))

            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(converted, lower_threshold, upper_threshold)

            # apply a series of erosions and dilations to the mask
            # using an elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            skinMask = cv2.erode(skinMask, kernel, iterations=3)
            skinMask = cv2.dilate(skinMask, kernel, iterations=3)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (11, 11), 5)
            skin = cv2.bitwise_and(frame, frame, mask=skinMask)
            ###################################################################
           # Parte para o corte da região da pele
            _, thresh = cv2.threshold(skinMask, 40, 255, 0)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                # draw in blue the contours that were founded
                cv2.drawContours(skin, contours, -1, (0, 255, 0), 1)

                # find the biggest countour (c) by the area
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
  # Return the Skin image
    return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR), x, y, w, h

## Section Two.2 : Function to remove black pixels from extracted image
def removeBlack(estimator_labels, estimator_cluster):
  
  
  # Check for black
  hasBlack = False
  
  # Get the total number of occurance for each color
  occurance_counter = Counter(estimator_labels)

  
  # Quick lambda function to compare to lists
  compare = lambda x, y: Counter(x) == Counter(y)
   
  # Loop through the most common occuring color
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    # Quick List comprehension to convert each of RBG Numbers to int
    color = [int(i) for i in estimator_cluster[x[0]].tolist() ]
    
  
    
    # Check if the color is [0,0,0] that if it is black 
    if compare(color , [0,0,0]) == True:
      # delete the occurance
      del occurance_counter[x[0]]
      # remove the cluster 
      hasBlack = True
      estimator_cluster = np.delete(estimator_cluster,x[0],0)
      break
      
   
  return (occurance_counter,estimator_cluster,hasBlack)
    
##Section Two.3 : Extract Colour Information
def getColorInformation(estimator_labels, estimator_cluster,hasThresholding=False):
  
  # Variable to keep count of the occurance of each color predicted
  occurance_counter = None
  
  # Output list variable to return
  colorInformation = []
  
  
  #Check for Black
  hasBlack =False
  
  # If a mask has be applied, remove th black
  if hasThresholding == True:
    
    (occurance,cluster,black) = removeBlack(estimator_labels,estimator_cluster)
    occurance_counter =  occurance
    estimator_cluster = cluster
    hasBlack = black
    
  else:
    occurance_counter = Counter(estimator_labels)
 
  # Get the total sum of all the predicted occurances
  totalOccurance = sum(occurance_counter.values()) 
  
 
  # Loop through all the predicted colors
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    index = (int(x[0]))
    
    # Quick fix for index out of bound when there is no threshold
    index =  (index-1) if ((hasThresholding & hasBlack)& (int(index) !=0)) else index
    
    # Get the color number into a list
    color = estimator_cluster[index].tolist()
    
    # Get the percentage of each color
    color_percentage= (x[1]/totalOccurance)
    
    #make the dictionay of the information
    colorInfo = {"cluster_index":index , "color": color , "color_percentage" : color_percentage }
    
    # Add the dictionary to the list
    colorInformation.append(colorInfo)
    
      
  return colorInformation 

## Section Two.4 : Putting it All together
def extractDominantColor(image,number_of_colors=5,hasThresholding=False):
  
  # Quick Fix Increase cluster counter to neglect the black(Read Article) 
  if hasThresholding == True:
    number_of_colors +=1
  # Taking Copy of the image
  img = image.copy()
  
  # Convert Image into RGB Colours Space
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  
  # Reshape Image
  img = img.reshape((img.shape[0]*img.shape[1]) , 3)
  
  #Initiate KMeans Object
  estimator = KMeans(n_clusters=number_of_colors, random_state=0)
  
  # Fit the image
  fit=estimator.fit(img)
  
  # Get Colour Information
  colorInformation = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
  return colorInformation,fit
  
##Section Two.4.1 : Putting it All together: Making a Visually Representation
def plotColorBar(colorInformation):
  #Create a 500x100 black image
  color_bar = np.zeros((100,500,3), dtype="uint8")
  
  top_x = 0
  for x in colorInformation:    
    bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

    color = tuple(map(int,(x['color'])))
  
    cv2.rectangle(color_bar , (int(top_x),0) , (int(bottom_x),color_bar.shape[0]) ,color , -1)
    top_x = bottom_x
  return color_bar,color

## Section Two.4.2 : Putting it All together: Pretty Print
def prety_print_data(color_info):
  for x in color_info:
    print(pprint.pformat(x))
    print()



#%%
##Section Three: Baking the Pie

# Get Image from URL. If you want to upload an image file and use that comment the below code and replace with  image=cv2.imread("FILE_NAME")

filename = 't8.mp4'
image = cv2.VideoCapture(filename)
success, frame = image.read()

skin,x, y, w, h= extractSkin(filename)         
cropeedIMAGE = frame[y:y+h, x:x+w]  
cv2.imshow('Image skin', cropeedIMAGE)
#plt.imshow(cv2.cvtColor(cropeedIMAGE,cv2.COLOR_BGR2RGB))
#plt.show()

# Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
dominantColors, fit= extractDominantColor(cropeedIMAGE,hasThresholding=True)
#fit
#Show in the dominant color information
#print("Color Information")
#prety_print_data(dominantColors)


#Show in the dominant color as bar
#print("Color Bar")
colour_bar,color = plotColorBar(dominantColors)
#plt.axis("off")
#plt.imshow(colour_bar)
#plt.show()
print("Color Information")
prety_print_data(color)