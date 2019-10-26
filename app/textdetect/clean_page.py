import numpy as np
import cv2

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        hull = cv2.convexHull(c)
        cv2.drawContours(temp_mask, [hull], 0, 255, -1)

    return temp_mask
