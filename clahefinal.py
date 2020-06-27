# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:16:14 2020

@author: Suranglikar
"""
import cv2
import glob

path = glob.glob("*.jpg") 
for img in path:
  id = img[:img.index('.')]
  image = cv2.imread(img,0)
  cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
  clahe = cl.apply(image)
  cv2.imwrite("Age Related/{}.jpg".format(id), clahe)
