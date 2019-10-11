#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
from collections import Counter
import pickle
import numpy as np

import app.textdetect.connected_components as cc
import app.textdetect.clean_page as clean
import app.textdetect.segmentation as seg

from app import app
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

def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False

def resize_and_pad(img, dheight, dwidth):
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = dwidth - new_size[1]
    delta_h = dheight - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

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

def crop_image_only_outside(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()-1
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()-1
    return img[row_start:row_end,col_start:col_end]

def k_nearest_neighbors(data, predict, k=3):
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

def segment_lines(image):
    result = []
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    #dilation
    kernel = np.ones((100,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)


    for ctr in sorted_ctrs:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if w < 20 or h < 20:
            continue

        # Getting ROI
        roi = image[y:y+h, x:x+w]
        roi = image_resize(roi, width=40)

        result.append(roi)

    return result

def ocr_boxes(image):
    lines = segment_lines(image)
    txt = u""
    for line in lines:
        #TODO: resize line of characters here
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

            if (h < 15) and ((i + 1) < (sizeOfChars-1)):
                while (h < 15) and ((i + 1) < (sizeOfChars-1)):
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

                    if characteris == "punctuation":
                        result = k_nearest_neighbors(dataset2, predict, k=3)

                    else:
                        result = k_nearest_neighbors(dataset, predict, k=11)
            except:
                result = u""

            txt += result

            # print result.encode("utf-8")
            # cv2.imshow("imgROIResized", imgROIResized)
            # cv2.waitKey(0)

    return txt

def detect_boxes():
    infile = os.path.join(app.config['UPLOAD_FOLDER'], "original_manga.jpg")
    img = cv2.imread(infile)
    img = image_resize(img, width=1075)
    gray = clean.grayscale(img)

    segmented_image = seg.segment_image(gray)
    segmented_image = segmented_image[:,:,2]

    #text columns are in the 3rd channel
    #columns = segmented_image[:,:,2]
    components = cc.get_connected_components(segmented_image)

    #perhaps do more strict filtering of connected components because sections of characters
    #will not be dropped from run length smoothed areas? Yes. Results quite good.
    #filtered = cc.filter_by_size(img,components,average_size*100,average_size*1)

    (h, w) = img.shape[:2]
    neww = 400
    resize = float(neww) / float(w)
    newh = int(resize * h)

    resized_img = cv2.resize(img, (neww, newh))

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "manga.jpg"), resized_img)

    blurbs = []
    id = 0
    for component in components:
        #TODO: ocr here

        minx = int(component[1].start * resize)
        miny = int(component[0].start * resize)
        maxx = int(component[1].stop * resize)
        maxy = int(component[0].stop * resize)

        roi = img[component[0].start:component[0].stop, component[1].start:component[1].stop]
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "textarea{}.jpg".format(id)), roi)

        jtext = ocr_boxes(roi)

        if not jtext: continue

        translation = translator.translate(jtext, dest='en')
        romaji = translator.translate(jtext, dest='ja')

        blurbs.append({"id": id, "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, 'jtext':jtext, 'romaji':romaji.pronunciation, 'etext':translation.text})
        id += 1

        print jtext.encode('utf-8')

    return blurbs