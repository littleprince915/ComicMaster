import numpy as np
import cv2

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adjust histogram to maximize black/white range (increase contrast, decrease brightness)??
    # gray = cv2.equalizeHist(gray)
    return gray


def binarize(img, threshold=190, white=255):
    (t, binary) = cv2.threshold(img, threshold, white, cv2.THRESH_BINARY_INV)
    return binary


def form_canny_mask(img, mask=None):
    edges = cv2.Canny(img, 128, 255, apertureSize=3)
    if mask is not None:
        mask = mask * edges
    else:
        mask = edges
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    temp_mask = np.zeros(img.shape, np.uint8)
    for c in contours:
        # also draw detected contours into the original image in green
        # cv2.drawContours(img,[c],0,(0,255,0),1)
        hull = cv2.convexHull(c)
        cv2.drawContours(temp_mask, [hull], 0, 255, -1)
        # cv2.drawContours(temp_mask,[c],0,255,-1)
        # polygon = cv2.approxPolyDP(c,0.1*cv2.arcLength(c,True),True)
        # cv2.drawContours(temp_mask,[polygon],0,255,-1)
    return temp_mask
