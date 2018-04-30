'''
  File name: estimateFeatureTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for single features 
    - Input startX: the x coordinate for single feature wrt the first frame
    - Input startY: the y coordinate for single feature wrt the first frame
    - Input Ix: the gradient along the x direction
    - Input Iy: the gradient along the y direction
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newX: the x coordinate for the feature wrt the second frame
    - Output newY: the y coordinate for the feature wrt the second frame
'''

import numpy as np
from numpy.linalg import solve, pinv, lstsq
from helper import plot_features, genEngMap
import matplotlib.pyplot as plt

WINDOW_SIZE = 3

def interp(data, nearX, nearY, xfrac, yfrac):
    XfYf = data[np.floor(nearX).astype(int),np.floor(nearY).astype(int)]
    XcYf = data[np.ceil(nearX).astype(int),np.floor(nearY).astype(int)]
    XfYc = data[np.floor(nearX).astype(int),np.ceil(nearY).astype(int)]
    XcYc = data[np.ceil(nearX).astype(int),np.ceil(nearY).astype(int)]

    XinterpYf = XfYf*(1-xfrac) + XcYf*(xfrac)
    XinterpYc = XfYc*(1-xfrac) + XcYc*(xfrac)
    interp = XinterpYf*(1-yfrac) + XinterpYc*(yfrac)
    return interp

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):

    In, In, Iprev = genEngMap(img1, sigma=1)
    In, In, Icurr = genEngMap(img2, sigma=1)
    It = Icurr-Iprev
#    plt.imshow(It)
#    plt.show()

    u = np.zeros(startX.shape)
    v = np.zeros(startY.shape)

    for i in range(startX.shape[0]):
        X = startX[i]
        Y = startY[i]

        nearX, nearY = np.meshgrid(np.arange(max(0,X-WINDOW_SIZE/2),min(img1.shape[0],X+WINDOW_SIZE/2+1)),np.arange(max(0,Y-WINDOW_SIZE/2),min(img1.shape[1],Y+WINDOW_SIZE/2+1)))

        # Interpolation
        xfrac = X-np.ceil(X)
        yfrac = Y-np.ceil(Y)
        nearIx = interp(Ix, nearX, nearY, xfrac, yfrac)
        nearIy = interp(Iy, nearX, nearY, xfrac, yfrac)
        nearIt = interp(It, nearX, nearY, xfrac, yfrac)
        #nearIx = Ix[nearX.astype(int),nearY.astype(int)]
        #nearIy = Iy[nearX.astype(int),nearY.astype(int)]
        #nearIt = It[nearX.astype(int),nearY.astype(int)]
        #plt.imshow(nearIx)
        #plt.show()

        A = np.zeros((2,2))
        A[0,0] = np.sum(np.multiply(nearIx,nearIx))
        A[0,1] = np.sum(np.multiply(nearIx,nearIy))
        A[1,0] = np.sum(np.multiply(nearIx,nearIy))
        A[1,1] = np.sum(np.multiply(nearIy,nearIy))

        B = np.zeros((2,1))
        B[0,0] = 0-np.sum(np.multiply(nearIx,nearIt))
        B[1,0] = 0-np.sum(np.multiply(nearIy,nearIt))

#        Ainv = pinv(A)
#        vu = np.dot(A,B)
        uv = lstsq(A,B)

        v[i] = uv[0][0,0]
        u[i] = uv[0][1,0]

    newX = startX + u#np.mean(u)
    newY = startY + v#np.mean(v)
    print "calculated newX and newY"
    return newX, newY
