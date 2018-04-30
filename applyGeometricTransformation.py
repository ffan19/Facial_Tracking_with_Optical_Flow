import numpy as np
from helper import ransac_est_homography, distance, est_homography

'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for bounding box
    - Input startXs: the x coordinates for all features wrt the first frame
    - Input startYs: the y coordinates for all features wrt the first frame
    - Input newXs: the x coordinates for all features wrt the second frame
    - Input newYs: the y coordinates for all features wrt the second frame
    - Input bbox: corner coordiantes of all detected bounding boxes

    - Output Xs: the x coordinates(after eliminating outliers) for all features wrt the second frame
    - Output Ys: the y coordinates(after eliminating outliers) for all features wrt the second frame
    - Output newbbox: corner coordiantes of all detected bounding boxes after transformation
'''

THRESHOLD = 4

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxes):

    N, faces = startXs.shape

    # Calculate all the distances at once
    dists = np.sqrt((newXs - startXs)**2 + (newYs - startYs)**2)
    # find the indices where distance > threshold
    toobig = np.where((dists > THRESHOLD) | (newXs < 0) | (newYs < 0))
    # create Xs and Ys
    Xs = newXs.copy()
    Ys = newYs.copy()
    # set X and Y to -1 at all the indices where distance > threshold
    Xs[toobig] = -1
    Ys[toobig] = -1

    newXs_noout = np.delete(newXs, toobig, 0)
    newYs_noout = np.delete(newYs, toobig, 0)
    startXs_noout = np.delete(startXs, toobig, 0)
    startYs_noout = np.delete(startYs, toobig, 0)

    # new boxes' corner coordinates
    newbboxes = np.zeros((faces, 4, 2))

    # For each face, reformat old bbox into 2 by 4 matrices, and take dot of it with homography
    for f in range(faces):
        print "f", f
        #print np.vstack((startXs_noout[:,f], startYs_noout[:,f], newXs_noout[:,f], newYs_noout[:,f])).T

        trynum = 0
        bboxdistance = 10000
        mindist = bboxdistance
        bestbbox = None
        while trynum < 10 and bboxdistance > 10:
            H = ransac_est_homography(startYs_noout[:,f], startXs_noout[:,f], newYs_noout[:,f], newXs_noout[:,f])
            curr_matrix = np.transpose(bboxes[f, :, :])
            third_row = np.ones((1, 4))
            curr_u = np.vstack([curr_matrix, third_row])
            # print "curr_u", curr_u
            curr_new_matrix = np.dot(H, curr_u)
            third_row = curr_new_matrix[2, :]
            # pprint "curr new matrix", curr_new_matrix
            final_curr_matrix = np.transpose(np.delete(curr_new_matrix, (2), axis=0) / third_row)
            # print "final curr matrix", final_curr_matrix
            bboxdistance = np.sum(np.sqrt((bboxes[f,:,:] - final_curr_matrix)**2))
            if bboxdistance < mindist:
                bestbbox = final_curr_matrix
                mindist = bboxdistance
                print "NEW MIN"
            trynum += 1
            print trynum, "BBOX DISTANCE", bboxdistance
        print "FINAL DISTANCE", mindist
        newbboxes[f, :, :] = bestbbox

    return Xs, Ys, newbboxes
