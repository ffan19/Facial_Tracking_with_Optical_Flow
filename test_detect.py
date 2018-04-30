import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

from detectFace import detectFace
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation
from helper import plot_features, plot_bboxes, plot_all, count_minfeats


def test_detect(vidfname, outfname):
    #vid1 = imageio.get_reader("shifted.mp4", 'ffmpeg')
    vid1 = imageio.get_reader(vidfname, 'ffmpeg')

    prevframe = None
    prevbboxes = None
    prevXs, prevYs = np.zeros((1,1)), np.zeros((1,1))
    allimg = []
    for i, frame in enumerate(vid1):
        print i
        if i % 30 == 0 or count_minfeats(prevXs) < 20 :
            print "RECALCULATING"
            try:
                newbboxes = detectFace(frame)
            except AttributeError:
                print "NO NEW FACE FOUND! Trying again on next frame"
                continue
            greyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Xs, Ys = getFeatures(greyframe, newbboxes)
            print Xs.shape
            prevXs, prevYs = Xs, Ys
        else:
            Xs, Ys = estimateAllTranslation(prevXs, prevYs, prevframe, frame)
            Xs, Ys, newbboxes = applyGeometricTransformation(prevXs, prevYs, Xs, Ys, prevbboxes)
            #plot_all(frame, newbboxes, Xs, Ys, display=True)

        allimg.append(plot_all(frame, newbboxes, Xs, Ys, display=False))
       # if i==0 or (i > 9 and not i % 10):
       # plot_features(frame, Xs, Ys, display=True)
       #     print Xs-prevXs

        prevbboxes = newbboxes
        prevframe = frame
        prevXs, prevYs = Xs.copy(), Ys.copy()

#        if i > 60:
#            print "Quitting early for testing purposes"
#            break

    imageio.mimsave(outfname, allimg)
    return


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "tkkagg":
        plt.switch_backend("tkagg")

#    test_detect("data/Easy/MarquesBrownlee.mp4", "VideoMarques_.mp4")
    test_detect("data/Easy/JonSnow.mp4", "VideoMartian_.mp4")
#    test_detect("data/Medium/TyrionLannister.mp4", "VideoTyrion_.mp4")
