import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/fifa_test_frame100.jpg', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(img, img, mask = mask)

plt.imshow(res[:,:,::-1])
plt.show()
