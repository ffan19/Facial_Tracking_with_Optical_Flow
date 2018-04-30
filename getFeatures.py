'''
  File name: getFeatures.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris

from helper import anms

FEATS_PER_FACE = 50

def getFeatures(img, bbox):
    cimg = corner_harris(img,sigma=1,k=0.1)
#    plt.imshow(cimg)
#    plt.show()
    x = np.zeros((FEATS_PER_FACE,bbox.shape[0]))
    y = np.zeros((FEATS_PER_FACE,bbox.shape[0]))

    for i, box in enumerate(bbox):
        bboxcimg = cimg*0
        boxx, boxy = np.meshgrid(np.arange(box[0,0], box[2,0]), np.arange(box[0,1], box[1,1]))
        bboxcimg[boxx,boxy] = cimg[boxx,boxy]
        fx, fy, rmax = anms(bboxcimg, FEATS_PER_FACE)
        x[:,i] = fx
        y[:,i] = fy

    return x, y
