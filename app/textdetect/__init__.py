#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import requests
import romkan

import clean_page as clean
import segmentation as seg
import connected_components as cc

from app.ocr.ocr import ocr_boxes
from app.ann import predict_ann

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
    cv2.imwrite("./processes/original.png", img_np)

    img = image_resize(img_np, 1075)
    cv2.imwrite("./processes/resized.png", img)

    gray = clean.grayscale(img)
    cv2.imwrite("./processes/grayscale.png", gray)

    segmented_image = seg.segment_image(gray)
    segmented_image = segmented_image[:, :, 2]

    components = cc.get_connected_components(segmented_image)

    (h, w) = img.shape[:2]
    neww = 400
    resize = float(neww) / float(w)
    newh = int(resize * h)

    resized_img = cv2.resize(img, (neww, newh))

    blurbs = extract_and_translate(components, img, resize)

    return resized_img, blurbs

def get_text_areas_npy(img_np):
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

    blurbs = extract_and_translate(components, img, resize)

    return resized_img, blurbs

def get_text_areas_without_ann(imagebytes):
    # img_np = cv2.imdecode(imagebytes, cv2.IMREAD_COLOR)
    img_np = cv2.imdecode(np.frombuffer(imagebytes, np.uint8), -1)
    cv2.imwrite("./processes/original.png", img_np)

    img = image_resize(img_np, 1075)
    cv2.imwrite("./processes/resized.png", img)

    gray = clean.grayscale(img)
    cv2.imwrite("./processes/grayscale.png", gray)

    segmented_image = seg.segment_image(gray)
    segmented_image = segmented_image[:, :, 2]

    components = cc.get_connected_components(segmented_image)

    (h, w) = img.shape[:2]
    neww = 400
    resize = float(neww) / float(w)
    newh = int(resize * h)

    resized_img = cv2.resize(img, (neww, newh))

    blurbs = []
    i = 0
    for component in components:
        i = i + 1
        minx = int(component[1].start * resize)
        miny = int(component[0].start * resize)
        maxx = int(component[1].stop * resize)
        maxy = int(component[0].stop * resize)

        roi = img[component[0].start:component[0].stop, component[1].start:component[1].stop]

        roidata = cv2.imencode('.jpg', roi)[1].tostring()

        blurbs.append({"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, "roidata": roidata})

    return resized_img, blurbs

def extract_and_translate(components, img, resize):
    blurbs = []
    sentences = []
    i = 0
    complen = len(components)
    for component in components:
        i = i+1
        minx = int(component[1].start * resize)
        miny = int(component[0].start * resize)
        maxx = int(component[1].stop * resize)
        maxy = int(component[0].stop * resize)

        roi = img[component[0].start:component[0].stop, component[1].start:component[1].stop]
        isjapanese, resized_textarea = predict_ann(roi)

        cv2.imwrite("./processes/textareas/textarea_{}.png".format(i), roi)
        cv2.imwrite("./processes/ANN/ann_{}.png".format(i), resized_textarea)


        if not isjapanese: continue

        roidata = cv2.imencode('.jpg', roi)[1].tostring()
        jtext = ocr_boxes(roi, i)


        if not jtext: continue
        sentences.append(jtext)
        # TODO: translating feature add here
        # translation = translator.translate(jtext, dest='en')
        # romaji = translator.translate(jtext, dest='ja')
        #
        # blurbs.append({"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, 'jtext': jtext,
        #                'romaji': romaji.pronunciation, 'etext': translation.text, "roidata": roidata})

        print 'line %i of %i: ' %(i, complen),
        print jtext
        etext, romaji = translate_sentences(jtext)
        # etext = [""]


        blurbs.append({"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, 'jtext': jtext,
                       'romaji': romaji, 'etext': etext, "roidata": roidata})

    # result = translate_sentences(sentences)
    # for x in xrange(len(blurbs)):
    #     blurbs[x]["etext"] = result[x]

    return blurbs


def get_text_areas_only(imagebytes):
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

    blurbs = extract_text_areas(components, img, resize)

    return resized_img, blurbs


def extract_text_areas(components, img, resize):
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
    return blurbs


def ann_classify(roibytes):
    roi = cv2.imdecode(np.frombuffer(roibytes, np.uint8), -1)
    isjapanese, _ = predict_ann(roi)


    return isjapanese


def get_characters(roibytes):
    roi = cv2.imdecode(np.frombuffer(roibytes, np.uint8), -1)
    jtext = ocr_boxes(roi)
    if jtext:
        print jtext
        etext, romaji = translate_sentences(jtext)
        return jtext, etext, romaji

    return "", "", ""

def translate_sentences(sentences):
    payload = {'text': sentences}
    r = requests.post("http://127.0.0.1:5001/translate", json=payload)
    result = r.json()
    return result["transformertranslate"], result["romaji"]