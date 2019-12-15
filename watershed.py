import numpy as np
import cv2
from matplotlib import colors as clr
import matplotlib.pylab as plb
import math
img = cv2.imread('rgb_01_02_009_04.png')
cv2.imshow("origin", img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('sure_fg', sure_bg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)

cmap = plb.cm.jet # define the colormap
# extract all colors from the .jet map
cmaplist = [clr.to_rgb(cmap(i)) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (0, 0, 0)
img[markers == -1] = [0, 0, 0]
for i in range(np.max(markers)):
    img[markers == i+1] = [math.floor(x*255) for x in
                           cmaplist[(len(cmaplist) // np.max(markers)) * i]]
cv2.imshow("result", img)
cv2.waitKey()
