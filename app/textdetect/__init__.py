#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

import clean_page as clean
import segmentation as seg
import connected_components as cc

from app.ocr.ocr import ocr_boxes
from app.ann import predict_ann

from googletrans import Translator

translator = Translator()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def get_text_areas(imagebytes):
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
        isjapanese = predict_ann(roi)

        if not isjapanese: continue

        roidata = cv2.imencode('.jpg', roi)[1].tostring()
        jtext = ocr_boxes(roi)

        if not jtext: continue

        translation = translator.translate(jtext, dest='en')
        romaji = translator.translate(jtext, dest='ja')

        blurbs.append({"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, 'jtext': jtext,
                       'romaji': romaji.pronunciation, 'etext': translation.text, "roidata": roidata})

    return resized_img, blurbs


def get_text_areas_only(imagebytes):
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

        roidata = cv2.imencode('.jpg', roi)[1].tostring()

        blurbs.append({"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, 'jtext': u"",
                       'romaji': u"", 'etext': u"", "roidata": roidata})

    return resized_img, blurbs


def ann_classify(roibytes):
    roi = cv2.imdecode(np.frombuffer(roibytes, np.uint8), -1)
    isjapanese = predict_ann(roi)

    return isjapanese


def get_characters(roibytes):
    roi = cv2.imdecode(np.frombuffer(roibytes, np.uint8), -1)
    jtext = ocr_boxes(roi)
    romaji = translator.translate(jtext, dest='ja')
    romaji = romaji.pronunciation

    return jtext, romaji
