import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt

frame = utils.read_image('images/fifa_test_frame100.jpg')
# frame = cv2.medianBlur(frame, 5)
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
sensitivity = 120
lower = np.array([0, 0, 255 - sensitivity])
upper = np.array([255, sensitivity, 255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower, upper)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

# gray = cv2.cvtColor(mask,cv2.COLOR_HSV2GRAY)
edges = cv2.Canny(mask,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

#cv2.imshow('frame',frame)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)

plt.imshow(frame)
plt.show()
