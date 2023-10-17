import numpy as np
import cv2
import run_length_smoothing as rls
import ocr

import connected_components as cc
import clean_page as clean


def segment_image(img, max_scale=4.0, min_scale=0.15):
    (h, w) = img.shape[:2]
    binary_threshold = 190

    img_binary = clean.binarize(img, threshold=binary_threshold)

    average_size = 30

    max_size = average_size * max_scale
    min_size = average_size * min_scale

    cleaned = cc.form_mask(img_binary, max_size, min_size, printing=True)

    text_only = cleaned2segmented(cleaned, average_size)

    if False:
        cv2.imshow("grayscale image", img)
        cv2.imshow("binary image", img_binary)
        cv2.imwrite("./processes/binary.png", img_binary)
        cv2.imshow("size filtered image", cleaned * 255)
        cv2.imwrite("./processes/sizefiltered.png", cleaned * 255)
        cv2.imshow("cleaned2segmented image", text_only)
        cv2.imwrite("./processes/boundingboxes.png", text_only)
        cv2.waitKey(0)

    text_like_areas = cc.get_connected_components(text_only)

    text_only = np.zeros(img.shape)
    cc.draw_bounding_boxes(text_only, text_like_areas, color=(255), line_size=-1)

    segmented_image = np.zeros((h, w, 3), np.uint8)
    segmented_image[:, :, 0] = img
    segmented_image[:, :, 1] = text_only
    segmented_image[:, :, 2] = text_only
    return segmented_image


def cleaned2segmented(cleaned, average_size):
    vertical_smoothing_threshold = 0.75 * average_size
    horizontal_smoothing_threshold = 0.75 * average_size
    (h, w) = cleaned.shape[:2]

    run_length_smoothed = rls.RLSO(cleaned, vertical_smoothing_threshold,
                                   horizontal_smoothing_threshold)
    components = cc.get_connected_components(run_length_smoothed)
    text = np.zeros((h, w), np.uint8)
    # text_columns = np.zeros((h,w),np.uint8)
    # text_rows = np.zeros((h,w),np.uint8)

    i = 0
    for component in components:
        i = i + 1
        seg_thresh = 1
        (aspect, v_lines, h_lines) = ocr.segment_into_lines(cleaned, component,
                                                            min_segment_threshold=seg_thresh)

        cv2.imwrite("./processes/CC_textareas/cc_{}.png".format(i), cleaned[component] * 255)
        print "cc_textarea_{}: ".format(i) + str(component) + "  vert_spaces: {}, hori_spaces: {}".format(len(v_lines), len(h_lines))

        if len(v_lines) < 2 and len(h_lines) < 2: continue

        ocr.draw_2d_slices(text, [component], color=255, line_size=-1)
        # ocr.draw_2d_slices(text_columns,v_lines,color=255,line_size=-1)
        # ocr.draw_2d_slices(text_rows,h_lines,color=255,line_size=-1)
    return text
