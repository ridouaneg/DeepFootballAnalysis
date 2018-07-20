import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/frame1296.jpg')
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
lower = np.array([0, 150, 0], dtype = "uint8")
upper = np.array([255, 255, 255], dtype = "uint8")
mask = cv2.inRange(hls, lower, upper)
res = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
res = cv2.cvtColor(res, cv2.COLOR_HLS2BGR)

"""plt.imshow(res, cmap = "gray")
plt.show()"""

"""img = res
edges = cv2.Canny(img,100,200,L2gradient=True)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()"""

height,width = mask.shape
skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
temp_nonzero = np.count_nonzero(mask)
while(np.count_nonzero(mask) != 0 ):
    eroded = cv2.erode(mask,kernel)
    temp = cv2.dilate(eroded,kernel)
    temp = cv2.subtract(mask,temp)
    skel = cv2.bitwise_or(skel,temp)
    mask = eroded.copy()
img = res
edges = cv2.Canny(skel, 50, 150)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=30,maxLineGap=30)
for i in range(len(lines)):
    x1,y1,x2,y2 = lines[i][0]
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    print(x1,y1,x2,y2)
"""
img = res
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200, apertureSize = 3)
lines = cv2.HoughLines(edges, 0.01, np.pi/180, 50)
for i in range(len(lines)):
    r, theta = lines[i][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)
"""
plt.imshow(img)
plt.show()
