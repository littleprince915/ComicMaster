import numpy as np
import cv2
import scipy.ndimage
import run_length_smoothing as rls
import ocr

import connected_components as cc
import clean_page as clean
import defaults


def segment_image(img, max_scale=defaults.CC_SCALE_MAX, min_scale=defaults.CC_SCALE_MIN):
    (h, w) = img.shape[:2]
    # create gaussian filtered and unfiltered binary images
    binary_threshold = defaults.BINARY_THRESHOLD
    binary = clean.binarize(img, threshold=binary_threshold)
    # binary_average_size = cc.average_size(binary)
    '''
    The necessary sigma needed for Gaussian filtering (to remove screentones and other noise) seems
    to be a function of the resolution the manga was scanned at (or original page size, I'm not sure).
    Assuming 'normal' page size for a phonebook style Manga is 17.5cmx11.5cm (6.8x4.5in).
    A scan of 300dpi will result in an image about 1900x1350, which requires a sigma of 1.5 to 1.8.
    I'm encountering many smaller images that may be nonstandard scanning dpi values or just smaller
    magazines. Haven't found hard info on this yet. They require sigma values of about 0.5 to 0.7.
    I'll therefore (for now) just calculate required (nonspecified) sigma as a linear function of vertical
    image resolution.
    '''
    sigma = (0.8 / 676.0) * float(h) - 0.9

    gaussian_filtered = scipy.ndimage.gaussian_filter(img, sigma=sigma)

    gaussian_binary = clean.binarize(gaussian_filtered, threshold=binary_threshold)

    # Draw out statistics on average connected component size in the rescaled, binary image
    average_size = cc.average_size(gaussian_binary)

    max_size = average_size * max_scale
    min_size = average_size * min_scale

    # primary mask is connected components filtered by size
    mask = cc.form_mask(gaussian_binary, max_size, min_size)

    # secondary mask is formed from canny edges
    canny_mask = clean.form_canny_mask(gaussian_filtered, mask=mask)

    # final mask is size filtered connected components on canny mask
    final_mask = cc.form_mask(canny_mask, max_size, min_size)

    # apply mask and return images
    cleaned = cv2.bitwise_not(final_mask * binary)
    text_only = cleaned2segmented(cleaned, average_size)

    if False:
        cv2.imshow("original image", img)
        cv2.imshow("thresholded image1", binary)
        cv2.imshow("gaussian filtered image", gaussian_filtered)
        cv2.imshow("thresholded image2", gaussian_binary)
        cv2.imshow("size filtered image", mask * 255)
        cv2.imshow("canny mask image", canny_mask)
        cv2.imshow("size filtered canny mask image", final_mask * 255)
        cv2.imshow("final_mask * binary image", cleaned)
        cv2.imshow("cleaned2segmented image", text_only)
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
    vertical_smoothing_threshold = defaults.VERTICAL_SMOOTHING_MULTIPLIER * average_size
    horizontal_smoothing_threshold = defaults.HORIZONTAL_SMOOTHING_MULTIPLIER * average_size
    (h, w) = cleaned.shape[:2]

    run_length_smoothed = rls.RLSO(cv2.bitwise_not(cleaned), vertical_smoothing_threshold,
                                   horizontal_smoothing_threshold)
    components = cc.get_connected_components(run_length_smoothed)
    text = np.zeros((h, w), np.uint8)
    # text_columns = np.zeros((h,w),np.uint8)
    # text_rows = np.zeros((h,w),np.uint8)

    for component in components:
        seg_thresh = 1
        (aspect, v_lines, h_lines) = ocr.segment_into_lines(cv2.bitwise_not(cleaned), component,
                                                            min_segment_threshold=seg_thresh)


        if len(v_lines) < 2 and len(h_lines) < 2: continue

        ocr.draw_2d_slices(text, [component], color=255, line_size=-1)
        # ocr.draw_2d_slices(text_columns,v_lines,color=255,line_size=-1)
        # ocr.draw_2d_slices(text_rows,h_lines,color=255,line_size=-1)
    return text
