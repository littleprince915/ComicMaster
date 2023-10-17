import numpy as np
import scipy.ndimage
from pylab import zeros, amax
import cv2
import pprint


def area_bb(a):
    return np.prod([max(x.stop - x.start, 0) for x in a[:2]])

def get_connected_components(image):
    labels, n = scipy.ndimage.measurements.label(image)
    objects = scipy.ndimage.measurements.find_objects(labels)
    return objects


def masks(image, connected_components, max_size, min_size):
    mask = zeros(image.shape, np.uint8)  # ,'B')
    for component in connected_components:
        size = area_bb(component) ** .5
        if size < min_size: continue
        if size > max_size: continue
        # a = area_nz(component,image)
        # if a<min_size: continue
        # if a>max_size: continue
        # print str(image[component])
        mask[component] = image[component] > 0
        # print str(mask[component])
    return mask


def draw_bounding_boxes(img, connected_components, max_size=0, min_size=0, color=(0, 0, 255), line_size=2):
    for component in connected_components:
        size = area_bb(component) ** .5
        if min_size > 0 and size < min_size: continue
        if max_size > 0 and size > max_size: continue
        # a = area_nz(component,img)
        # if a<min_size: continue
        # if a>max_size: continue
        (ys, xs) = component[:2]
        cv2.rectangle(img, (xs.start, ys.start), (xs.stop, ys.stop), color, line_size)


def average_size(img, minimum_area=3, maximum_area=100):
    components = get_connected_components(img)
    sorted_components = sorted(components, key=area_bb)
    # sorted_components = sorted(components,key=lambda x:area_nz(x,binary))
    areas = zeros(img.shape)
    for component in sorted_components:
        # As the input components are sorted, we don't overwrite
        # a given area again (it will already have our max value)
        if amax(areas[component]) > 0: continue
        # take the sqrt of the area of the bounding box
        areas[component] = area_bb(component) ** 0.5
        # alternate implementation where we just use area of black pixels in cc
        # areas[component]=area_nz(component,binary)
    # we lastly take the median (middle value of sorted array) within the region of interest
    # region of interest is defaulted to those ccs between 3 and 100 pixels on a side (text sized)
    aoi = areas[(areas > minimum_area) & (areas < maximum_area)]
    if len(aoi) == 0:
        return 0
    return np.median(aoi)


def form_mask(img, max_size, min_size, printing=False):
    components = get_connected_components(img)
    sorted_components = sorted(components, key=area_bb)

    i = 0
    for component in sorted_components:
        i = i + 1
        cv2.imwrite("./processes/CC/cc_{}.png".format(i), img[component])
        print "cc_{}: ".format(i) + str(component)

    mask = masks(img, sorted_components, max_size, min_size)
    return mask
