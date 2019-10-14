#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

import clean_page as clean
import segmentation as seg
import connected_components as cc

from app.ocr.ocr import ocr_boxes

from googletrans import Translator
translator = Translator()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_text_areas(imagebytes):
    # img_np = cv2.imdecode(imagebytes, cv2.IMREAD_COLOR)
    img_np = cv2.imdecode(np.frombuffer(imagebytes, np.uint8), -1)
    img = image_resize(img_np, 1075)

    gray = clean.grayscale(img)

    segmented_image = seg.segment_image(gray)
    segmented_image = segmented_image[:, :, 2]

    components = cc.get_connected_components(segmented_image)

    (h, w) = img.shape[:2]
    neww = 400
    resize = float(neww) / float(w)
    newh = int(resize * h)

    resized_img = cv2.resize(img, (neww, newh))

    blurbs = []
    for component in components:
        minx = int(component[1].start * resize)
        miny = int(component[0].start * resize)
        maxx = int(component[1].stop * resize)
        maxy = int(component[0].stop * resize)

        roi = img[component[0].start:component[0].stop, component[1].start:component[1].stop]

        #TODO: call ANN function that evaluates if text area is japanese or not

        roidata = cv2.imencode('.jpg', roi)[1].tostring()
        jtext = ocr_boxes(roi)

        if not jtext: continue

        translation = translator.translate(jtext, dest='en')
        romaji = translator.translate(jtext, dest='ja')

        blurbs.append({"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, 'jtext': jtext,
                       'romaji': romaji.pronunciation, 'etext': translation.text, "roidata":roidata})

    return resized_img, blurbs
