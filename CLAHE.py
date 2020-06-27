import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Dataset/Training/Diabetes/90_left.jpg", 0)
equ = cv2.equalizeHist(img)

plt.hist(equ.flat, bins=100, range=(0,100))

cv2.imshow("Original Image", img)
cv2.imshow("Equalized", equ)

cv2.waitKey(0)          
cv2.destroyAllWindows() 
