#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from collections import Counter
from PIL import ImageFont, ImageDraw, Image
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


global dataset2

desired_size = 26
iteration = 0
hori_iteration = 0
global_character_number = 0
def create_dataset():
    img = np.ones((200, 200, 3), np.uint8) * 255
    b, g, r, a = 0, 0, 0, 0
    fontpaths = ["./fonts/hiragino mincho w6.otf", "./fonts/heisei-mincho-std-w5.otf",
                 "./fonts/heisei-mincho-std-w7.otf", "./fonts/heisei-mincho-std-w9.otf",
                 "./fonts/hiragino-kaku-gothic-pro-w3.otf", "./fonts/hiragino-kaku-gothic-pro-w6.otf",
                 "./fonts/hiragino-kaku-gothic-std-w8.otf",
                 "./fonts/hiragino-mincho-pro-w3.otf", "./fonts/meiryo.ttc", "./fonts/ms gothic.TTF",
                 "./fonts/ms mincho.ttf", "./fonts/quimi mincho.ttf"]
    jpn_chars = u"ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべ" + \
                u"ぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセ" + \
                u"ゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴ" + \
                u"日一国会人年大十二本中長出三同時政事自行社見月分議後前民生連五発間対上部東者党地合市業内相方四定今回新場金員九入選" + \
                u"立開手米力学問高代明実円関決子動京全目表戦経通外最言氏現理調体化田当八六約主題下首意法不来作性的要用制治度務強気小" + \
                u"七成期公持野協取都和統以機平総加山思家話世受区領多県続進正安設保改数記院女初北午指権心界支第産結百派点教報済書府活" + \
                u"原先共得解名交資予川向際査勝面委告軍文反元重近千考判認画海参売利組知案道信策集在件団別物側任引使求所次水半品昨論計" + \
                u"死官増係感特情投示変打男基私各始島直両朝革価式確村提運終挙果西勢減台広容必応演電住争談能無再位置真流格有疑口過局少" + \
                u"放税検町常校料裁状工建語球営空職証土急止送供可役構木割聞身費付切由説転食比難防補車優夫研収断何南石足消境神番規術護" + \
                u"展態導備宅害配副算視条幹独警宮究育席輸訪楽起万着乗店述残想線率病農州武声質念待試族象銀域助労例衛然早張映限親額験追" + \
                u"商葉義伝働形景落担好退準賞辺造英株頭技低毎医復仕去姿味負閣失移差衆個門写評課末守若脳極種美命福蔵量望松非観察整段横" + \
                u"型白深字答夜製票音申様財港識注呼達良帰針専推谷古候史天階程満敗管値歌買兵接器士光討路悪科授細効図週積丸他録処省旧室汚曜"

    dataset = {}
    dataset2 = {}
    tmp = 0

    jpn_chars = list(jpn_chars)


    jpn_chars.append(u"!?")
    jpn_chars.append(u"!!")
    jpn_chars.append(u"!")
    jpn_chars.append(u"?")

    for char in jpn_chars:
        dataset[char] = []
        dataset2[char] = []

        for fontpath in fontpaths:
            font = ImageFont.truetype(fontpath, 64)

            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((20, 20), char, font=font, fill=(b, g, r, a))

            char_img = np.array(img_pil)

            imgGray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            imgBlurred = cv2.GaussianBlur(imgGray, (15, 15), 0)
            imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            imgROI = crop_image_only_outside(imgThresh)
            imgROIResized = resize_and_pad(imgROI, desired_size, desired_size)

            # cv2.imwrite("./charsdataset/{}.png".format(tmp), imgROIResized)

            dataset2[char].append(imgROIResized)
            dataset[char].append(imgROIResized.reshape((1, desired_size * desired_size)))
            tmp += 1
            if tmp % 100 == 0: print "KNN dataset creating: {}".format(tmp)


    return dataset

def recreate_dataset():
    dataset = create_dataset()

    picklefile = open("dataset.pickle", "wb")
    pickle.dump(dataset, picklefile)
    picklefile.close()

    global dataset2
    dataset2 = dataset.copy()
    dataset.pop(u"?", None)
    dataset.pop(u"!", None)
    dataset.pop(u"!?", None)
    dataset.pop(u"!!", None)

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
            euclidean_distance = np.linalg.norm(np.array(features, dtype="float32") - np.array(predict, dtype="float32"))
            distances.append([euclidean_distance, group, features])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    # if vote_result == u"ん":
    #     distances = sorted(distances)[:k]
    #     num = 0
    #     for dist in distances:
    #         d = dist[2].reshape((desired_size, desired_size))
    #         fig, ax = plt.subplots(figsize=(7, 7))
    #
    #         fig.tight_layout()
    #         ax.set_xticks(np.arange(desired_size))
    #         ax.set_yticks(np.arange(desired_size))
    #         ax.set_title("Distance: {:.2f}".format(dist[0]), fontdict={'fontsize': 18})
    #         im = ax.imshow(d, cmap=cm.gray)
    #
    #         for i in range(desired_size):
    #             for j in range(desired_size):
    #                 if d[i, j] > 128: color = "black"
    #                 else: color = "white"
    #                 text = ax.text(j, i, d[i, j], ha="center", va="center", color=color, fontsize="x-small")
    #         # plt.show()
    #         plt.savefig('./figure_{}.png'.format(num))
    #         num += 1
    #
    #     d = predict.reshape((desired_size, desired_size))
    #     fig, ax = plt.subplots(figsize=(7, 7))
    #
    #     fig.tight_layout()
    #     ax.set_xticks(np.arange(desired_size))
    #     ax.set_yticks(np.arange(desired_size))
    #     ax.set_title("Converted Character Image", fontdict={'fontsize': 18})
    #     im = ax.imshow(d, cmap=cm.gray)
    #
    #     for i in range(desired_size):
    #         for j in range(desired_size):
    #             if d[i, j] > 128:
    #                 color = "black"
    #             else:
    #                 color = "white"
    #             text = ax.text(j, i, d[i, j], ha="center", va="center", color=color, fontsize="x-small")
    #     # plt.show()
    #     plt.savefig('./predict.png')

    return vote_result

def segment_lines(image):
    global iteration
    iteration = iteration + 1
    result = []
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    #dilation
    kernel = np.ones((100,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imwrite("./processes/OCR_vertical/textarea_{}.png".format(iteration), image)
    cv2.imwrite("./processes/OCR_vertical/vertdil_{}.png".format(iteration), img_dilation)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)

    c = 0
    for ctr in sorted_ctrs:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if w < 20 or h < 20:
            continue

        # Getting ROI
        roi = image[y:y+h, x:x+w]
        roi = image_resize(roi, width=40)

        c = c+1
        cv2.imwrite("./processes/OCR_vertical/line_{}_{}.png".format(iteration, c), roi)

        result.append(roi)

    return result

def ocr_boxes(image, box_number=0):
    lines = segment_lines(image)
    txt = u""
    global global_character_number
    line_number = 0

    for line in lines:
        line_number = line_number + 1
        imgThresh, sorted_ctrs = dilate_horizontally_and_threshold(line)

        chars = group_pixels_by_char(imgThresh, sorted_ctrs)

        char_number = 0
        for char in chars:
            char_number = char_number + 1
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

                    # if box_number == 7:
                    #     cv2.imwrite("./processes/OCR_chars/char_{}_{}.png".format(line_number, char_number), imgROI)

                    imgROI = crop_image_only_outside(imgROI)
                    # if box_number == 7:
                    #     cv2.imwrite("./processes/OCR_cropped/char_{}_{}.png".format(line_number, char_number), imgROI)

                    imgROIResized = resize_and_pad(imgROI, desired_size, desired_size)
                    # if box_number == 7:
                    #     cv2.imwrite("./processes/OCR_padded/char_{}_{}.png".format(line_number, char_number), imgROIResized)

                    cv2.imwrite("./processes/OCR_padded/char_{}.jpg".format(global_character_number), imgROIResized)
                    global_character_number+=1

                    predict = imgROIResized.reshape((1, desired_size * desired_size))

                    result = k_nearest_neighbors(dataset2, predict, k=7)

            except:
                result = u""

            txt += result

    return txt

def dilate_horizontally_and_threshold(line):
    global hori_iteration
    hori_iteration = hori_iteration + 1
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((1, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imwrite("./processes/OCR_horizontal/line_{}.png".format(hori_iteration), line)
    cv2.imwrite("./processes/OCR_horizontal/dilated_{}.png".format(hori_iteration), img_dilation)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    imgGray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    filenumber = 0
    for ctr in sorted_ctrs:
        filenumber = filenumber + 1
        x, y, w, h = cv2.boundingRect(ctr)
        roi = imgThresh[y:y + h, x:x + w]
        cv2.imwrite("./processes/OCR_pixels/pixels_{}_{}.png".format(hori_iteration, filenumber), roi)


    return imgThresh, sorted_ctrs

def group_pixels_by_char(imgThresh, sorted_ctrs):
    chars = []
    i = 0

    sizeOfChars = len(sorted_ctrs)
    while i < sizeOfChars:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(sorted_ctrs[i])
        characteris = "character"

        if (h < 15) and ((i + 1) < sizeOfChars):
            while (h < 15) and ((i + 1) < sizeOfChars):
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
                        try:
                            nx, ny, nw, nh = cv2.boundingRect(sorted_ctrs[i + 1])
                            h = h + nh + (ny - (h + y))
                            i += 1
                            characteris = "ellipsis"
                        except:
                            print "an error happened"

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
    return chars


# picklefile = open("dataset.pickle", "rb")
# dataset = pickle.load(picklefile)
# picklefile.close()

# recreate_dataset()