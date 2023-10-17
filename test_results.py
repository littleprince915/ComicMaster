#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app import app
from app.ocr.ocr import recreate_dataset
from app.ann import create_ann_dataset
from app.textdetect import get_text_areas, get_text_areas_only, get_text_areas_npy, ann_classify, get_characters
import cv2
import glob
import numpy as np

create_ann_dataset()
recreate_dataset()

i = 1
for filename in glob.glob("./tdtest_dataset/*.jpg"):
    image = np.array(cv2.imread(filename))
    image, textareas = get_text_areas_npy(image)

    overlay = image.copy()
    for textarea in textareas:
        cv2.rectangle(overlay, (textarea["minx"], textarea["miny"]), (textarea["maxx"], textarea["maxy"]), (0, 255, 0), -1)


    alpha = 0.5
    image_new = cv2.addWeighted(image, alpha, overlay, 1-alpha, 0)

    cv2.imwrite("./tdtest_results/result_{}.jpg".format(i), image_new)
    i += 1