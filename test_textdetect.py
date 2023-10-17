import glob
import cv2
import numpy as np

from app.textdetect import get_text_areas_npy

i = 0
for filename in glob.glob("./images/*.jpg"):
    image = np.array(cv2.imread(filename))
    image, textareas = get_text_areas_npy(image)
    overlay = image.copy()
    print "processing {}".format(i)
    for t in textareas:
        cv2.rectangle(overlay, (t["minx"], t["miny"]), (t["maxx"], t["maxy"]), (0, 255, 0), -1)

    alpha = 0.5
    image_new = cv2.addWeighted(image, alpha, overlay, 1-alpha, 0)

    # cv2.imshow("result", image_new)
    # cv2.waitKey()

    cv2.imwrite("./results/result_{}.jpg".format(i), image_new)
    i += 1
