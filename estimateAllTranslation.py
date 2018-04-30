'''
  File name: estimateAllTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for all features for each bounding box as well as its four corners
    - Input startXs: all x coordinates for features wrt the first frame
    - Input startYs: all y coordinates for features wrt the first frame
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newXs: all x coordinates for features wrt the second frame
    - Output newYs: all y coordinates for features wrt the second frame
'''

import numpy as np
import matplotlib.pyplot as plt

from estimateFeatureTranslation import estimateFeatureTranslation
from helper import genEngMap

def estimateAllTranslation(startXs, startYs, img1, img2):

    newXs = np.zeros(startXs.shape)
    newYs = np.zeros(startYs.shape)


    for f in range(startXs.shape[1]):
        startX = startXs[:,f]
        startY = startYs[:,f]

        Iy, Ix, Ixy1 = genEngMap(img1,sigma=1)
#        Ix2, Iy2, Ixy2 = genEngMap(img2,sigma=10)
#        Ix = (Ix1 + Ix2)/2
#        Iy = (Iy1 + Iy2)/2
#        plt.imshow(Ix)
#        plt.show()
        newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
        newXs[:,f] = newX
        newYs[:,f] = newY

    return newXs, newYs
