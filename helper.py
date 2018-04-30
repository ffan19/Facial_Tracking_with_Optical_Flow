'''
  File name: helper.py
  Author:
  Date created:
'''

'''
  File clarification:
  Include any helper function you want for this project such as the 
  video frame extraction, video generation, drawing bounding box and so on.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from math import sqrt
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
import pdb, random


def count_minfeats(coords_all):
    minfeats = coords_all.shape[0]
    for f in range(coords_all.shape[1]):
        coords = coords_all[:,f]
        minfeats = min(minfeats, np.count_nonzero(coords >= 0))
    print "MIN FEATS", minfeats
    return minfeats


def distance((x1, y1), (x2, y2)):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def plot_all(img, bboxes, Xs, Ys, display=True):
    fig, ax = plt.subplots(1)
    canvas = FigureCanvas(fig)
    ax.imshow(img)
    for f, rawbbox in enumerate(bboxes):
        bbox = np.zeros((4, 2))
        bbox[0, 0] = rawbbox[3, 1]
        bbox[0, 1] = rawbbox[3, 0]
        bbox[1, 0] = rawbbox[2, 1]
        bbox[1, 1] = rawbbox[2, 0]
        bbox[2, 0] = rawbbox[0, 1]
        bbox[2, 1] = rawbbox[0, 0]
        bbox[3, 0] = rawbbox[1, 1]
        bbox[3, 1] = rawbbox[1, 0]

        rect = patches.Polygon(bbox, linewidth=2, edgecolor="magenta", facecolor="none", closed=True)
        ax.add_patch(rect)

    for i in range(Xs.shape[1]):
        X = Xs[:, i]
        Y = Ys[:, i]
        for xx, yy in zip(X, Y):
            circ = plt.Circle((yy, xx), 5, color='cyan', fill=False)
            ax.add_patch(circ)
    if display:
        plt.show()
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plotimg = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close(fig)
    return plotimg


def plot_bboxes(img, bboxes):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for rawbbox in bboxes:
        bbox = np.zeros((4, 2))
        bbox[0, 0] = rawbbox[3, 1]
        bbox[0, 1] = rawbbox[3, 0]
        bbox[1, 0] = rawbbox[2, 1]
        bbox[1, 1] = rawbbox[2, 0]
        bbox[2, 0] = rawbbox[0, 1]
        bbox[2, 1] = rawbbox[0, 0]
        bbox[3, 0] = rawbbox[1, 1]
        bbox[3, 1] = rawbbox[1, 0]

        rect = patches.Polygon(bbox, linewidth=3, edgecolor="red", facecolor="none", closed=True)
        ax.add_patch(rect)
    plt.show()


def plot_features(img, Xs, Ys, display=True):
    fig, ax = plt.subplots(1)
    canvas = FigureCanvas(fig)
    ax.set_aspect('equal')
    ax.imshow(img)
    for i in range(Xs.shape[1]):
        X = Xs[:, i]
        Y = Ys[:, i]
        for xx, yy in zip(X, Y):
            circ = plt.Circle((yy, xx), 5, color='cyan', fill=False)
            ax.add_patch(circ)
    if display:
        plt.show()
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plotimg = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close(fig)
    return plotimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def genEngMap(I, sigma):
    dim = I.ndim
    if dim == 3:
        Ig = rgb2gray(I)
    else:
        Ig = I

    Ig = Ig.astype(np.float64())

    [gradx, grady] = np.gradient(gaussian_filter(Ig, sigma=sigma))
    gradxy = gradx + grady
    return gradx, grady, gradxy


def anms(cimg, max_pts):

    x = []
    y = []
    rmax = 0
    pointsbyval = defaultdict(lambda: set())
    valsbypoint = {}
    mindistbypoint = {}
    lessthanneighbor = set()

    # ignore low value points
    threshquant = 100 - 100.0 * max_pts / cimg.size * 10
#    print "anms threshold\t", threshquant
    lowthresh = np.percentile(cimg, threshquant)
#    plt.imshow(cimg > lowthresh)
#    ID = input("ID for corner: ")
#    ID = "0_2"
#    plt.imsave("corners"+str(ID)+".png",cimg > lowthresh)
#    plt.show()

#    print lowthresh

    slack = 0.99

    for j in range(21, cimg.shape[1] - 21):
        for i in range(21, cimg.shape[0] - 21):
            val = cimg[i, j]
            if val < lowthresh:
                continue
            if (i - 1, j) in valsbypoint:
                if valsbypoint[(i - 1, j)] * slack > val:
                    lessthanneighbor.add((i, j))
            if (i, j - 1) in valsbypoint:
                if valsbypoint[(i, j - 1)] * slack > val:
                    lessthanneighbor.add((i, j))
            valsbypoint[(i, j)] = val
            pointsbyval[val].add((i, j))
            mindistbypoint[(i, j)] = cimg.shape[0] + cimg.shape[1]

#    print "feature pts\t", max_pts, "\tselected from\t", len(pointsbyval)

    for pt, val in valsbypoint.iteritems():
        # skip some of the ones which are less than their neighbor
        if pt in lessthanneighbor:
            del mindistbypoint[pt]
            continue
        for otherval, otherpts in pointsbyval.iteritems():
            if val < otherval * slack:
                for otherpt in otherpts:
                    dist = distance(pt, otherpt)
                    if dist < mindistbypoint[pt]:
                        mindistbypoint[pt] = dist
                        if dist < 1.4:
                            break

    sortedpts = sorted(mindistbypoint.iteritems(), key=lambda (k, v): (v, k), reverse=True)[0:max_pts]
    rmax = sortedpts[-1][-1]

    for rank, (pt, dist) in enumerate(sortedpts):
        # print pt, dist, valsbypoint[pt]
        x.append(pt[0])
        y.append(pt[1])

#    plot_features(cimg, x, y)

    return np.asarray(x).astype(int), np.asarray(y).astype(int), rmax


def est_homography(x2d, y2d, X2d, Y2d):

    x = np.ndarray.flatten(x2d)
    y = np.ndarray.flatten(y2d)
    X = np.ndarray.flatten(X2d)
    Y = np.ndarray.flatten(Y2d)

    N = x.size
    A = np.zeros([2 * N, 9])

    i = 0
    while i < N:
        a = np.array([x[i], y[i], 1]).reshape(-1, 3)
        c = np.array([[X[i]], [Y[i]]])
        d = - c * a

        A[2 * i, 0: 3], A[2 * i + 1, 3: 6] = a, a
        A[2 * i: 2 * i + 2, 6:] = d

        i += 1

    # compute the solution of A
    U, s, V = np.linalg.svd(A, full_matrices=True)
    h = V[8, :]
    H = h.reshape(3, 3)

    return H


def ransac_est_homography(x1, y1, x2, y2):

    THRESH = 0.5

    def compute_ssd(xy1, xyt):
        ssd = (xy1-xyt)**2
        return np.sqrt(ssd[0]**2 + ssd[1]**2)


    def count_inliers(ssd, THRESH):
        return np.where(ssd < THRESH)[0].shape[0]


    nRANSAC = 3000

    maxinliers = 0

    for i in range(nRANSAC):
        randos = random.sample(range(0,x1.shape[0]),4)
        x1samp = x1[randos]
        y1samp = y1[randos]
        x2samp = x2[randos]
        y2samp = y2[randos]
        H = est_homography(y1samp,x1samp,y2samp,x2samp)
#        H = est_homography(x1,y1,x1,y1)
        xy2, xyt = apply_homography(H, x1, y1, x2, y2)
        ssd = compute_ssd(xy2, xyt)
        numinliers = count_inliers(ssd, THRESH)
        if numinliers > maxinliers:
            maxinliers = numinliers
            xbest1 =  x1samp.copy()
            ybest1 =  y1samp.copy()
            xbest2 =  x2samp.copy()
            ybest2 =  y2samp.copy()
    
    Hfinal = est_homography(ybest1,xbest1,ybest2,xbest2)
    xy2, xyt = apply_homography(Hfinal, x1, y1, x2, y2)
    ssd = compute_ssd(xy2, xyt)
    numinliers = count_inliers(ssd, THRESH)
#    print ssd, ssd < THRESH
    print "HOMOGRAPHY", x1.shape[0], numinliers
    inlier_ind = np.asarray(np.where(ssd < THRESH))

    return Hfinal


def apply_homography(H, x1, y1, x2, y2):

    z1 = y1/y1.astype(int)
    xyz1 = np.asarray(zip(y1, x1, z1)).T
    xy2 = np.asarray(zip(y2, x2)).T
    xyzt = np.dot(H, xyz1)

    xy1 = xyz1[0:2,:]
    xyt = xyzt[0:2,:]

    return xy2, xyt/xyzt[2,:]
