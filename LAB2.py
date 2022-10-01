# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:57:19 2022

@author: Juan Pablo
"""
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Import the cameraman image
cameraman = cv.imread('standard_test_images/cameraman.tif')
cameraman_g = cv.cvtColor(cameraman, cv.COLOR_BGR2GRAY)

#Import the house image
house = cv.imread('standard_test_images/house.tif')
house_g = cv.cvtColor(house, cv.COLOR_BGR2GRAY)

#Import the livingroom image
liv = cv.imread('standard_test_images/livingroom.tif')
liv_g = cv.cvtColor(liv, cv.COLOR_BGR2GRAY)

#Show the image
cv.imshow('House',house_g)
cv.waitKey(0)

#Camera man
#Histogram visualization
gray_hist = cv.calcHist([cameraman_g],[0],None,[256],[0,256])
#plt.hist(cameraman.ravel(),256,[0,256]); plt.show()

#Histogram equalization
gray_eq = cv.equalizeHist(cameraman_g)
equ = cv.equalizeHist(cameraman_g)
'''hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
'''

#Gaussian blur 5x5
blur = cv.GaussianBlur(cameraman, (5,5),2)
#cv.imshow('Gaussian',blur)
#cv.waitKey(0)

#Display camera man results
'''
plt.figure()
plt.title('Histogram visualization')
plt.xlabel('Bins')
plt.ylabel('No. Pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
'''
'''
plt.figure()
plt.title('Original')
plt.subplot(1, 2, 1)
plt.plot(cameraman)

plt.title('Gaussian blur 5x5')
plt.subplot(1, 2, 2)
plt.plot(blur)

plt.show()
'''


#House
#Sobel edge detection
house_blur = cv.GaussianBlur(house_g,(5,5),2)
#sobelx = cv.Sobel(src=house, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge
#cv.imshow('Sobel X Y using Sobel() function', sobelx)
#cv.waitKey(0)
'''
def sobel(src):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad
grad = sobel(house)
cv.imshow('Sobel edge detection', grad)
cv.waitKey(0)
'''
#Harris corner detector
'''
img = house
gray = np.float32(house_g)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
'''
#Canny edge detector
'''
img = house_g
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
cv.imshow('Canny',edges)
cv.waitKey(0)
'''
#Hough transform (straight lines)
'''
image1 = house
gray=cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
dst = cv.Canny(gray, 50, 200)

lines= cv.HoughLines(dst, 1, math.pi/180.0, 100, np.array([]), 0, 0)

a,b,c = lines.shape
for i in range(a):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0, y0 = a*rho, b*rho
    pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
    pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
    cv.line(image1, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)


cv.imshow('image1',image1)
cv.waitKey(0)
cv.destoryAllWindows(0)
'''

#Living room
#SIFT feature detection

img = liv
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imshow('SIFT',img)
cv.waitKey(0)