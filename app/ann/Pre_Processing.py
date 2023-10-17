#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:58:20 2019

@author: Enyang
"""

import glob
import numpy as np
import cv2
import re

def resize_and_pad(img, num_px):
    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratio = float(num_px) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = num_px - new_size[1]
    delta_h = num_px - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def load_dataset(subset_selection, fileType, num_px):
    if re.search(subset_selection, "training, evaluation, validation") is None:
        raise SystemExit("ERROR: Please select training, evaluation or validation")
    if fileType != 'jpg':
        raise SystemExit("ERROR: Please select JPG only")
    train_set_x = []
    train_set_y = []
    folder_name = './annimages/' + subset_selection
    folder_file_type_selection = folder_name + '/*/*.' + fileType
    print("INFO: Start to load dataset")
    tmp = 0
    for filename in glob.glob(folder_file_type_selection):  # assuming jpg
        image = np.array(cv2.imread(filename))
        try:
            my_image = resize_and_pad(image, num_px).reshape((num_px * num_px * 3,)).T
        except ValueError as e:
            print("Value Error: " + str(e))
            continue
        train_set_x.append(my_image)
        if filename.startswith(folder_name + '\japanese'):
            train_set_y.append(1)
        else:
            train_set_y.append(0)

        tmp += 1
        if tmp % 100 == 0: print "ANN dataset creating: {}".format(tmp)

    np_train_set_x = np.array(train_set_x).T
    print(np_train_set_x.shape)
    np_train_set_y = np.array(train_set_y).reshape(1, len(train_set_y))

    assert np_train_set_x.shape[0] == num_px * num_px * 3, "An image should have " + str(
        num_px * num_px * 3) + " pixels!"
    assert np_train_set_y.shape[0] == 1, "y should have shape (1, ..)"
    print(subset_selection + " dataset loaded successfully.")
    return np_train_set_x, np_train_set_y


def preprocess_data(image, num_px):
    train_set_x = []

    resized_image = resize_and_pad(image, num_px)

    data = resized_image.reshape((num_px * num_px * 3,)).T
    train_set_x.append(data)
    np_train_set_x = np.array(train_set_x).T

    return np_train_set_x, resized_image

#np_result_set_x, np_result_set_y = load_dataset('training','jpg',200)