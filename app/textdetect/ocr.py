import numpy as np
import cv2


def draw_2d_slices(img, slices, color=(0, 0, 255), line_size=1):
    for entry in slices:
        vert = entry[0]
        horiz = entry[1]
        cv2.rectangle(img, (horiz.start, vert.start), (horiz.stop, vert.stop), color, line_size)


def segment_into_lines(img, component, min_segment_threshold=1):
    (ys, xs) = component[:2]
    w = xs.stop - xs.start
    h = ys.stop - ys.start
    aspect = float(w) / float(h)

    vertical = []
    start_col = xs.start
    for col in range(xs.start, xs.stop):
        count = np.count_nonzero(img[ys.start:ys.stop, col])
        if count <= min_segment_threshold or col == (xs.stop):
            if start_col >= 0:
                vertical.append((slice(ys.start, ys.stop), slice(start_col, col)))
                start_col = -1
        elif start_col < 0:
            start_col = col

    horizontal = []
    start_row = ys.start
    for row in range(ys.start, ys.stop):
        count = np.count_nonzero(img[row, xs.start:xs.stop])
        if count <= min_segment_threshold or row == (ys.stop):
            if start_row >= 0:
                horizontal.append((slice(start_row, row), slice(xs.start, xs.stop)))
                start_row = -1
        elif start_row < 0:
            start_row = row

    return (aspect, vertical, horizontal)
