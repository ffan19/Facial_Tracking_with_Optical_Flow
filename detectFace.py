import numpy as np
import cv2
import matplotlib.pyplot as plt

from helper import plot_bboxes

CASCADE = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")

def detectFace(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))


    F = faces.shape[0]
    bbox = np.zeros((F,4,2))
    for i, face in enumerate(faces):
        minx = face[1]
        miny = face[0]
        xlen = face[3]
        ylen=face[2]
        bbox[i,0:2,0] = minx
        bbox[i,2:4,0] = minx+xlen
        bbox[i,(0,2),1] = miny
        bbox[i,(1,3),1] = miny+ylen
        
#    plot_bboxes(img, bbox)

    return bbox.astype(int)
