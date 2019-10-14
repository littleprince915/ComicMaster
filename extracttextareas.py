#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import glob

import app.textdetect.clean_page as clean
import app.textdetect.segmentation as seg
import app.textdetect.connected_components as cc


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


folder = "./images/*/*.jpg"
id = 0
for filename in glob.glob(folder):
    img = cv2.imread(filename)
    img = image_resize(img, width=1075)
    gray = clean.grayscale(img)

    segmented_image = seg.segment_image(gray)
    segmented_image = segmented_image[:,:,2]

    components = cc.get_connected_components(segmented_image)
    for component in components:
        roi = img[component[0].start:component[0].stop, component[1].start:component[1].stop]
        height, width, channels = roi.shape

        if width < 20 or height < 20:
            continue

        cv2.imwrite("./textareas/{}.jpg".format(id), roi)
        id += 1