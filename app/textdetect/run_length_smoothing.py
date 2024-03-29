import cv2


def vertical_run_length_smoothing(img, v_threshold):
    vertical = img.copy()
    (rows, cols) = vertical.shape
    # print "total rows " + str(rows) + " total cols "+ str(cols)
    for row in xrange(rows):
        for col in xrange(cols):
            value = vertical.item(row, col)
            if value == 0: continue
            next_row = row + 1
            while True:
                if next_row >= rows: break
                if vertical.item(next_row, col) > 0 and next_row - row <= v_threshold:
                    for n in range(row, next_row):
                        vertical.itemset(n, col, 255)
                    break
                if next_row - row > v_threshold: break
                next_row = next_row + 1
    return vertical


def horizontal_run_length_smoothing(img, h_threshold):
    horizontal = img.copy()
    (rows, cols) = horizontal.shape
    # print "total rows " + str(rows) + " total cols "+ str(cols)
    for row in xrange(cols):
        for col in xrange(rows):
            value = horizontal.item(col, row)
            if value == 0: continue
            # print "row : " + str(row) + " col: " + str(col)
            next_row = row + 1
            while True:
                if next_row >= cols: break
                if horizontal.item(col, next_row) > 0 and next_row - row <= h_threshold:
                    for n in range(row, next_row):
                        horizontal.itemset(col, n, 255)
                        # horizontal[col,n]=255
                    break
                    # print 'setting white'
                    # binary[row,col]=255
                if next_row - row > h_threshold: break
                next_row = next_row + 1
    return horizontal


def RLSO(img, h_threshold, v_threshold):
    horizontal = horizontal_run_length_smoothing(img, h_threshold)
    vertical = vertical_run_length_smoothing(img, v_threshold)
    run_length_smoothed_or = cv2.bitwise_or(vertical, horizontal)

    if False:
        cv2.imshow("horizontal rls", horizontal)
        cv2.imwrite("./processes/horizontalrls.png", horizontal)
        cv2.imshow("vertical rls", vertical)
        cv2.imwrite("./processes/verticalrls.png", vertical)
        cv2.imshow("vertical + horizontal rls", run_length_smoothed_or)
        cv2.imwrite("./processes/combinedrls.png", run_length_smoothed_or)

    return run_length_smoothed_or
