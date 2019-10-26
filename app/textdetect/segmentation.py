import numpy as np
import cv2
import scipy.ndimage
import run_length_smoothing as rls
import ocr

import connected_components as cc
import clean_page as clean
import defaults


def segment_image(img, max_scale=4.0, min_scale=0.15):
    (h, w) = img.shape[:2]

    binary_threshold = 190
    binary = clean.binarize(img, threshold=binary_threshold)

    sigma = (0.8 / 676.0) * float(h) - 0.9

    gaussian_filtered = scipy.ndimage.gaussian_filter(img, sigma=sigma)

    gaussian_binary = clean.binarize(gaussian_filtered, threshold=binary_threshold)

    average_size = cc.average_size(gaussian_binary)

    max_size = average_size * max_scale
    min_size = average_size * min_scale

    mask = cc.form_mask(gaussian_binary, max_size, min_size)

    canny_mask = clean.form_canny_mask(gaussian_filtered, mask=mask)

    final_mask = cc.form_mask(canny_mask, max_size, min_size)

    cleaned = cv2.bitwise_not(final_mask * binary)
    text_only = cleaned2segmented(cleaned, average_size)

    text_like_areas = cc.get_connected_components(text_only)

    text_only = np.zeros(img.shape)
    cc.draw_bounding_boxes(text_only, text_like_areas, color=(255), line_size=-1)

    segmented_image = np.zeros((h, w, 3), np.uint8)
    segmented_image[:, :, 0] = img
    segmented_image[:, :, 2] = text_only
    return segmented_image


def cleaned2segmented(cleaned, average_size):
    vertical_smoothing_threshold = defaults.VERTICAL_SMOOTHING_MULTIPLIER * average_size
    horizontal_smoothing_threshold = defaults.HORIZONTAL_SMOOTHING_MULTIPLIER * average_size
    (h, w) = cleaned.shape[:2]

    run_length_smoothed = rls.RLSO(cv2.bitwise_not(cleaned), vertical_smoothing_threshold,
                                   horizontal_smoothing_threshold)
    components = cc.get_connected_components(run_length_smoothed)
    text = np.zeros((h, w), np.uint8)

    for component in components:
        seg_thresh = 1
        (aspect, v_lines, h_lines) = ocr.segment_into_lines(cv2.bitwise_not(cleaned), component,
                                                            min_segment_threshold=seg_thresh)

        if len(v_lines) < 2 and len(h_lines) < 2: continue

        ocr.draw_2d_slices(text, [component], color=255, line_size=-1)
    return text
