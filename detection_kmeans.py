import cv2
import numpy as np
import matplotlib.pyplot as plt

t1 = cv2.getTickCount()

img = cv2.imread('images/fifa_test_frame100.jpg', 1)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print(time)

print(center)

cv2.namedWindow('res2', cv2.WINDOW_NORMAL)
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
