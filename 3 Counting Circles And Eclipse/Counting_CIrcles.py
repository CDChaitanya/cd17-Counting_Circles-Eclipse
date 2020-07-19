# COunting Circles And Eclipse

import numpy as np
import cv2

# Load Image
image = cv2.imread('blobs.jpg' , 0) # 0 FOR GRAY SCALE IMAGE
cv2.imshow('ORIGINAL IMAGE', image)
cv2.waitKey(0)

# INITIALIZE THE DETECTOR USING THE DEFAULT PARAMETERS
detector = cv2.SimpleBlobDetector_create()

# DETECT BLOBS
keypoints = detector.detect(image)

# DRAW BLOBS ON OUR IMAGE AS RED CIRCLE
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image=image, keypoints=keypoints, outImage=blank, 
                          color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

no_of_blobs = len(keypoints)
text = 'TOTAL NO. OF BLOBS: '+str(len(keypoints))
cv2.putText(img=blobs, text=text, org=(20,550), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=(100,0,255), thickness=2 )

# DISPLAY IMAGES WITH BLOB KEYPOINTS
cv2.imshow('BLOB (USING DEFAULT PARA)', blobs)
cv2.waitKey(0)

# SET YOUR FILTERING PARAMETER
# IMITIALIZE PARAMETER SETTING USING cv2.SimpleBlobDetector_Params()
para = cv2.SimpleBlobDetector_Params()

# SETTING AREA FILTERING PARAMETERS
para.filterByArea = True
para.minArea = 100

# SETTING CIRCULARITY FILTERING PARAMETERS
para.filterByCircularity = True
para.minCircularity = 0.9

# SETTING CONVEXITY FILTERING PARAMETERS
para.filterByConvexity = True
para.minConvexity = 0.2

# SETTING INERTIA FILTERING PARAMETERS
para.filterByInertia = True
para.minInertiaRatio = 0.01

# CREATE DETECTOR WITH THE PARAMETERS
detector = cv2.SimpleBlobDetector_create(para)

# DETECT BLOBS (THIS TIME ONLY CIRCLES )
keypoints = detector.detect(image)

# DRAW BLOBS ON OUR IMAGE AS RED CIRCLE
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image=image, keypoints=keypoints, outImage=blank, 
                          color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

no_of_blobs = len(keypoints)
text = 'TOTAL NO. OF CIRCULAR BLOBS: '+str(len(keypoints))
cv2.putText(img=blobs, text=text, org=(20,550), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=(0,100,255), thickness=2 )

# DISPLAY IMAGES WITH BLOB KEYPOINTS
cv2.imshow('BLOB (USING DEFAULT PARA)', blobs)
cv2.waitKey(0)

# SHOW BLOBS 
cv2.imshow('FILTERING CIRCULAR BLOBS ONLY' , blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()









