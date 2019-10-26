#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from collections import Counter
import pickle
import numpy as np

from googletrans import Translator

picklefile = open("dataset.pickle", "rb")
dataset = pickle.load(picklefile)
picklefile.close()

dataset2 = dataset.copy()
dataset.pop(u"?", None)
dataset.pop(u"!", None)
dataset.pop(u"!?", None)
dataset.pop(u"!!", None)

translator = Translator()
desired_size = 50


def resize_and_pad(img, dheight, dwidth):
    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = dwidth - new_size[1]
    delta_h = dheight - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


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


def crop_image_only_outside(img, tol=0):
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax() - 1
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax() - 1
    return img[row_start:row_end, col_start:col_end]


def k_nearest_neighbors(data, predict, k=3):
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features, dtype="float32") - np.array(predict, dtype="float32"))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


def segment_lines(image):
    result = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((100, 1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)

    for ctr in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        if w < 20 or h < 20:
            continue

        roi = image[y:y + h, x:x + w]
        roi = image_resize(roi, width=40)

        result.append(roi)

    return result


def ocr_boxes(image):
    lines = segment_lines(image)
    txt = u""

    for line in lines:
        # TODO: resize line of characters here
        gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # dilation
        kernel = np.ones((1, 100), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        # find contours
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

        imgGray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGray, (7, 7), 0)
        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

        chars = []
        i = 0
        sizeOfChars = len(sorted_ctrs)
        while i < sizeOfChars:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(sorted_ctrs[i])
            characteris = "character"

            if (h < 15) and ((i + 1) < sizeOfChars):
                while (h < 15) and ((i + 1) < sizeOfChars):
                    px, py, pw, ph = cv2.boundingRect(sorted_ctrs[i - 1])
                    nx, ny, nw, nh = cv2.boundingRect(sorted_ctrs[i + 1])

                    if (y - (py + ph)) > 20 and (ny - (y + h)) > 20:
                        break

                    else:
                        i += 1
                        h = h + nh + (ny - (h + y))
                        imgROI = imgThresh[y:y + h, x:x + w]
                        imgROI = crop_image_only_outside(imgROI)
                        height, width = imgROI.shape[:2]

                        if width < 7:
                            nx, ny, nw, nh = cv2.boundingRect(sorted_ctrs[i + 1])
                            h = h + nh + (ny - (h + y))
                            i += 1
                            characteris = "ellipsis"

            else:
                if (i + 1) == (sizeOfChars - 1):
                    nx, ny, nw, nh = cv2.boundingRect(sorted_ctrs[i + 1])

                    if nh < 17:
                        i += 1
                        h = h + nh + (ny - (h + y))
                        characteris = "punctuation"

                else:
                    imgROI = imgThresh[y:y + h, x:x + w]
                    imgROI = crop_image_only_outside(imgROI)
                    height, width = imgROI.shape[:2]

                    if (height < 7) and (width < 7) and (i == (sizeOfChars - 1)):
                        characteris = "period"

                    if width < 12:
                        characteris = "dash"

                    elif width < 15:
                        characteris = "tilde"

            chars.append([x, y, w, h, characteris])
            i += 1

        for char in chars:
            [intX, intY, intW, intH, characteris] = char
            try:
                if characteris == "ellipsis":
                    result = u"…"

                elif characteris == "tilde":
                    result = u"~"

                elif characteris == "dash":
                    result = u"ー"

                elif characteris == "period":
                    result = u"。"

                else:
                    imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
                    imgROI = crop_image_only_outside(imgROI)

                    imgROIResized = resize_and_pad(imgROI, desired_size, desired_size)

                    predict = imgROIResized.reshape((1, desired_size * desired_size))

                    result = k_nearest_neighbors(dataset, predict, k=5)

            except:
                result = u""

            txt += result

    return txt
