#  MIT License
#
#  Copyright (c) 2021.  Antoine Legrain <antoine.legrain@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# Reference:
# https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

# import the necessary packages
import sys
import os
from colorama import Fore, Style
import re
import numpy as np, cv2, imutils
import pandas as pd
from keras.models import load_model
from pdf2image import convert_from_path
from PIL import Image
from fpdf import FPDF


allowed_decimals = ['0', '25', '5', '75']
len_mat = 7
re_mat = '[1-2]\\d{6}'
RED = (225,6,0)
GREEN = (0,154,23)
BLACK=(0,0,0)


def grade_all_exams(path, grades_csv, box, dir_path='', classifier=None, dpi=300, margin=1):
    # loading our CNN model if needed
    if classifier is None:
        classifier = load_model('digit_recognizer.h5')

    ph = int(11 * dpi)
    pw = int(8.5*dpi)
    half_dpi = int(.5 * dpi)
    quarter_dpi = int(.25 * dpi)

    def get_blank_page(h=ph, w=pw, dim=None):
        if dim:
            return np.full((h, w, dim), 255, np.uint8)
        else:
            return np.full((h, w), 255, np.uint8)

    # # f = '/Users/legraina/Dropbox (MAGI)/Enseignement/Poly/MTH1102_Calcul_II/Correction du final/Groupe D_1/MTH1102D AUTOMNE 2021 GROUPE 01-4.pdf'
    # # grays = gray_images(f)
    # grays = [cv2.cvtColor(cv2.imread('images/page_%d.png' % d), cv2.COLOR_BGR2GRAY) for d in range(23)]
    # mat = find_matricule(grays[1:], box['matricule'], classifier)
    # try:
    #     total_matched, numbers, img = grade(grays[0], box['front']['grade'], classifier, add_border=True)
    #     if numbers: print("%s: %.2f" % (mat, numbers[-1]))
    #     # grades_df.at[mat, 'Note'] = numbers[-1]
    #     else: print(Fore.GREEN + "%s: No valid grade" % mat + Style.RESET_ALL)

    # list files and directories
    pmargin = int(margin*dpi)
    max_h = 0
    max_w = 0
    sumarries = []
    for f in os.listdir(path):
        pf = os.path.join(path, f)

        # if directory, grade files inside
        if os.path.isdir(pf):
            gcsv = grades_csv
            if not grades_csv.endswith('.csv'):
                gcsv = grades_csv+f+"_"
            grade_all_exams(pf, gcsv, box, dir_path+("/" if dir_path else "")+f, classifier)
            continue

        # grade files
        if not dir_path or not f.endswith('.pdf'):
            continue

        file = os.path.join(path, f)
        if os.path.isfile(file):
            grays = gray_images(file)
            mat, id_box = find_matricule(grays, box['front']['id'], box['matricule'], classifier)
            if not mat:
                print("No matricule found for %s" % f)

            total_matched, numbers, grades = grade(grays[0], box['front']['grade'], classifier, add_border=True)
            if numbers:
                print("%s: %.2f" % (f, numbers[-1]))

            # put everything in an image
            w = id_box.shape[1] + grades.shape[1] + dpi
            if w > max_w:
                max_w = w
            h = max(id_box.shape[0]+dpi, grades.shape[0])
            if h > max_h:
                max_h = h
            summary = get_blank_page(h, w)
            # add id
            summary[0:id_box.shape[0], 0:id_box.shape[1]] = id_box
            # add grades
            summary[0:grades.shape[0], id_box.shape[1]+dpi:w] = grades
            # write matricule and grade in color
            color_summary = cv2.cvtColor(summary, cv2.COLOR_GRAY2RGB)
            cv2.putText(color_summary,  "Matricule: "+(mat if mat else 'N/A'),
                        (int(2.5*dpi), id_box.shape[0]+half_dpi),  # position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, 2,  GREEN if mat else RED, 5)
            cv2.putText(color_summary, str(numbers[-1]) if numbers else 'N/A',
                        (id_box.shape[1]+half_dpi, h-half_dpi),  # position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN if total_matched else RED, 5)
            cv2.putText(color_summary, "%s: %s" % (dir_path, f),
                        (quarter_dpi, h-quarter_dpi),  # position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
            imwrite_png('summary', color_summary)
            sumarries.append(color_summary)
            if len(sumarries) > 4:
                break

    hmargin = int(pw - max_w) // 2  # horizontal margin
    imgh = max_h+15
    n_s = ph // imgh  # number of pictures by page
    vmargin = int(ph - n_s*imgh) // 2

    pages = []
    page = get_blank_page(dim=3)
    y = vmargin
    # put summaries on pages
    for s in sumarries:
        # new page if needed
        if y + s.shape[0] > ph - vmargin:
            imwrite_png("page", page)
            pages.append(Image.fromarray(page))
            page = get_blank_page(dim=3)
            y = vmargin
        # add summarry
        page[y:y+s.shape[0], hmargin:hmargin+s.shape[1]] = s
        y += s.shape[0]+5  # update cursor
        page[y:y+2, :] = BLACK
        y += 7
    imwrite_png("page", page)
    pages.append(Image.fromarray(page))

    # save pdf
    pages[0].save(grades_csv+"summary.pdf", save_all=True, append_images=pages[1:])


def grade_all(path, grades_csv, box):
    # load csv
    grades_df = pd.read_csv(grades_csv, index_col='Matricule')

    # loading our CNN model
    classifier = load_model('digit_recognizer.h5')

    # f = 'Thibault Emile_15482717_2082734_assignsubmission_file_Thibault_Emile_2082734_Devoir6_MTH1102_A21_Gr02.pdf'
    # # f2 = 'Bibombe Nathan_15482710_2118644_assignsubmission_file_Bibombe_Nathan_2118644_Devoir6_MTH1102_AUTOMNE2021_Gr02.pdf'
    # file = os.path.join(path, f)
    # numbers = grade(file, box, classifier)

    # grade files
    for f in os.listdir(path):
        if not f.endswith('.pdf'):
            continue
        # search matricule
        m = re.search(re_mat+'(?=\\D)', f)
        if not m:
            print("Matricule wasn't found in "+f)
        m = int(m.group())

        file = os.path.join(path, f)
        if os.path.isfile(file):
            gray = gray_images(file, [0])[0]
            total_matched, numbers, img = grade(gray, box, classifier)
            if numbers:
                print("%s: %.2f" % (f, numbers[-1]))
                grades_df.at[m, 'Note'] = numbers[-1]
            else:
                print(Fore.GREEN + "%s: No valid grade" % f + Style.RESET_ALL)
    # store grades
    grades_df.to_csv(grades_csv)


def compare_all(path, grades_csv, box):
    # load csv
    grades_df = pd.read_csv(grades_csv, index_col='Matricule')

    # loading our CNN model
    classifier = load_model('digit_recognizer.h5')

    # grade files
    tp = 0
    tpp = 0
    fp = 0
    fpp = 0
    fn = 0
    n = 0
    for f in os.listdir(path):
        if not f.endswith('.pdf'):
            continue
        # search matricule
        m = re.search('[1-2]\\d{6}(?=\\D)', f)
        if not m:
            print("Matricule wasn't found in "+f)
        m = int(m.group())

        file = os.path.join(path, f)
        if os.path.isfile(file):
            gray = gray_images(file, [0])[0]
            total_matched, numbers = grade(gray, box, classifier)
            if numbers:
                print("%s: %.2f" % (f, numbers[-1]))
                if grades_df.at[m, 'Note'] == numbers[-1]:
                    if total_matched:
                        tp += 1
                    else:
                        tpp += 1
                elif total_matched:
                    fp += 1
                else:
                    fpp += 1
                grades_df.at[m, 'Note'] = numbers[-1]
            else:
                print(Fore.GREEN + "%s: No valid grade" % f + Style.RESET_ALL)
                fn += 1
                grades_df.at[m, 'Note'] = -1
            n += 1
    # store grades
    print("Accuracy: %.3f (%.3f, %.3f), Precision: %.3f (%.3f, %.3f)"
          % ((tp+tpp) / n, tp / n, tpp / n, (tp+tpp) / (tp+tpp+fp+fpp), tp / (tp+fp), tpp / (tpp+fpp)))


def grade(gray, box, classifier=None, add_border=False):
    cropped = fetch_box(gray, box)
    boxes = find_grade_boxes(cropped, add_border)
    all_numbers = []
    for b in boxes:
        (x, y, w, h) = cv2.boundingRect(b)
        box_img = cropped[y + 5:y + h - 5, x + 5:x + w - 5]
        all_numbers.append(test(box_img.copy(), classifier))

    if len(all_numbers) == 0:
        print("No valid number has been found")
        return False, None, cropped

    # find all combination that works
    combinations = [(0, [])]
    for numbers in all_numbers:
        if len(numbers) == 0:
            print("No valid number has been found for at least one of the box")
            return False, None, cropped
        c2 = [(c+p, l+[i]) for p, i in numbers for c, l in combinations]
        combinations = c2
    combinations = sorted(combinations, reverse=True)
    for p, numbers in combinations:
        # if only one number, return it
        if len(numbers) <= 1:
            return True, numbers, cropped
        # check the sum
        total = sum(numbers[:-1])
        if total != numbers[-1]:
            continue
        return True, numbers, cropped

    # Has not been able to check the total -> give the best prediction
    expected_numbers = [n[0] for n in all_numbers[:-1]]
    p, total = (sum(n[0] for n in expected_numbers) / len(expected_numbers),
                sum(n[1] for n in expected_numbers))
    pt, nt = all_numbers[-1][0]
    return False, [n for p, n in expected_numbers[:-1]] + [nt if pt >= p else total], cropped


def gray_images(fpdf, pages=None, dpi=300, straighten=True):
    # fpdf: path to pdf file to grade
    images = []
    if pages is None:
        images = convert_from_path(fpdf, dpi=dpi)
    else:
        for p in pages:
            images.append(convert_from_path(fpdf, dpi=dpi, last_page=p+1, first_page=p))
    gray_images = []
    for i, img in enumerate(images):
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        if straighten:
            try:
                gray = imstraighten(gray)
            except ValueError as e:
                print("For page %d: %s" % (i, str(e)))
        imwrite_png("page_%d" % i, gray)
        gray_images.append(gray)
    return gray_images


def fetch_box(img, box):
    # box = (x1, x2, y1, y2) in %
    x = [int(box[0] * img.shape[1]), int(box[1] * img.shape[1]),
         int(box[2] * img.shape[0]), int(box[3] * img.shape[0])]
    cropped = img[x[2]:x[3], x[0]:x[1]]  # ys and then xs
    imwrite_png("cropped", cropped)
    return cropped


def imstraighten(gray):
    ngray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(ngray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    imwrite_png("thresh", thresh)
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    center, dim, angle = cv2.minAreaRect(coords)
    if angle > 45:
        angle = 90-angle
    if abs(angle) > 10:
        raise ValueError("Current page is too skewed.")
    # rotate the image to deskew it
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    # divide angle by 2 in case of error as we are changing the center and we are just using small angles
    M = cv2.getRotationMatrix2D(center, angle/2, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    imwrite_png("rotated", rotated)
    return rotated


def find_edges(cropped, thick=5, min_lenth=80, max_gap=15, angle_resolution=np.pi/2, line_on_original=False):
    """ Find the lines on the image. """
    # Computing the edge map via the Canny edge detector
    edged = cv2.Canny(cropped, 50, 200, 255)
    imwrite_png("canny", edged)
    edged = cv2.dilate(edged, kernel=np.ones((3, 3), dtype='uint8'))
    imwrite_png("dilated", edged)
    lines = cv2.HoughLinesP(edged, 1, angle_resolution, 50, minLineLength=min_lenth, maxLineGap=max_gap)
    if line_on_original:
        edged = cropped.copy()
    if lines is not None:
        for l in lines:
            cv2.line(edged, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255, 255, 255), thick)
        imwrite_png("edged", edged)

    # erode the image to keep only the big lines
    if not line_on_original:
        edged = cv2.erode(edged, kernel=np.ones((5, 5), dtype='uint8'))
        imwrite_png("eroded", edged)
    return edged


def find_grade_boxes(cropped, add_border=False, max_diff=50, thick=5):
    """ Find boxes on the image. """
    # add border: useful if having only partial boxes
    cropped2 = cropped.copy()
    if add_border:
        cv2.rectangle(cropped2, (thick, thick), (cropped2.shape[1]-thick, cropped2.shape[0]-thick),
                      (0, 0, 0), thick)
        imwrite_png("cropped", cropped2)

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts, hierarchy = cv2.findContours(find_edges(cropped2, thick=thick), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours((cnts, hierarchy))
    imwrite_contours("cropped_all_boxes", cropped2, cnts, thick=thick)

    # keep only the children of the biggest contour
    pos = max(enumerate(cnts), key=lambda cnt: cv2.contourArea(cnt[1]))[0]
    ccnts = biggest_children(cnts, hierarchy, pos)
    # Retrieve
    # loop over the contours to take the ones that are aligned with the biggest one and are big enough
    cropped2 = cropped.copy()
    boxes = []
    ref = None
    horizontal = None
    for c in sorted(ccnts, key=cv2.contourArea, reverse=True):
        (x, y, w, h) = cv2.boundingRect(c)
        # set the reference box
        if ref is None:
            ref = (x, y, w, h)
        # box too small
        elif w * h < .1 * ref[2] * ref[3]:
            break
        # for horizontal alignment
        elif abs(ref[1] - y) <= max_diff:
            if horizontal is None:
                horizontal = True
            # break the alignment
            elif not horizontal:
                continue
            # break the box size
            if abs(ref[3] - h) >= max_diff:
                continue
        # for vertical alignment
        elif abs(ref[0] - x) <= max_diff:
            if horizontal is None:
                horizontal = False
            # break the alignment
            elif horizontal:
                continue
            # break the box size
            elif abs(ref[2] - w) >= max_diff:
                continue
        # break alignment
        else:
            continue
        boxes.append(c)

    # sort boxes according to alignment
    boxes = sorted(boxes, key=lambda b: cv2.boundingRect(b)[0 if horizontal else 1])
    # remove the extreme boxes close to the border if has added some borders
    if add_border and boxes:
        (x, y, w, h) = cv2.boundingRect(boxes[0])
        if x + y <= 4*thick + 5:
            boxes = boxes[1:]
        (x, y, w, h) = cv2.boundingRect(boxes[-1])
        if x + y + h + w >= cropped.shape[0] + cropped.shape[1] - 4*thick - 5:
            boxes = boxes[:-1]
    # add any missing box
    if boxes:
        prev = None
        boxes2 = []
        for b in boxes:
            (x, y, w, h) = cv2.boundingRect(b)
            if prev is not None:
                if horizontal:
                    # add a contour
                    if x - prev > 4 * thick:
                        boxes2.append(np.array([[prev+thick, y-thick], [x-thick, y+h+thick]]))
                elif y - prev > 4 * thick:  # add a contour
                    boxes2.append(np.array([[x-thick, prev+thick], [x+w+thick, y-thick]]))
            boxes2.append(b)
            prev = x+w if horizontal else y+h
        boxes = boxes2
    imwrite_contours("cropped_boxes", cropped2, boxes, thick=2*thick, padding=-thick)
    return boxes


def find_matricule(grays, box_id, box_mat, classifier):
    possible_digits = [{} for i in range(len_mat)]

    def find_digits(gray_box, split=False):
        # find contours of the numbers as well as the dot number position
        # return a sorted list of the relevant digits' contours and the dot position (and the threshold image used)
        # 10 = len("Matricule:")
        cnts, dot, thresh = find_digit_contours(gray_box, max_cnts=len_mat,
                                                split_on_semi_column=split, min_box_before_split=10)
        # check length
        if len(cnts) != len_mat:
            return False
        # extract digits
        all_digits = extract_all_digits(cnts, gray_box, thresh, classifier)
        # store values
        for i, d_cnt in enumerate(all_digits):
            distri = possible_digits[i]
            for p, d in d_cnt[1]:
                if d in distri:
                    distri[d] += p
                else:
                    distri[d] = p
        return True

    # find the id box
    cropped = fetch_box(grays[0], box_id)
    cnts, hierarchy = cv2.findContours(find_edges(cropped), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours((cnts, hierarchy))
    imwrite_contours("rgray", cropped, cnts, thick=5)
    # Find the biggest contour for the id box
    pos, biggest_c = max(enumerate(cnts), key=lambda cnt: cv2.contourArea(cnt[1]))
    id_box = get_image_from_contour(cropped, biggest_c)
    for cnt in biggest_children(cnts, hierarchy, pos):
        if find_digits(get_image_from_contour(cropped, cnt), True):
            break

    # try to find a matricule on the next page
    for gray in grays[1:]:
        cropped = fetch_box(gray, box_mat)
        mgray = find_edges(cropped, thick=3, line_on_original=True, max_gap=5, min_lenth=150)
        find_digits(mgray)

    # build matricules and sort them by probabilities
    matricules = [(0, '')]
    for distri in possible_digits:
        matricules = [(c+p, '%s%d' % (m, d)) for c, m in matricules for d, p in distri.items()]
    smats = sorted(matricules, reverse=True)

    # find the most valid and probable one
    for p, mat in smats:
        if re.match(re_mat, mat):
            return mat, id_box

    return None, id_box


def test(gray_img, classifier=None):
    if classifier is None:
        classifier = load_model('digit_recognizer.h5')

    # image copy
    gray = gray_img.copy()
    imwrite_png("gray", gray)

    # find contours of the numbers as well as the dot number position
    # return a sorted list of the relevant digits' contours and the dot position (and the threshold image used)
    cnts, dot, thresh = find_digit_contours(gray)

    # extract digits
    all_digits = extract_all_digits(cnts, gray, thresh, classifier)

    if not all_digits:
        print("No valid number has been found")
        return []

    # process all possible digits combinations
    return process_digits_combinations(all_digits, dot)


def get_image_from_contour(img, cnt):
    (x, y, w, h) = cv2.boundingRect(cnt)
    img = img[y:y+h, x:x+w]
    imwrite_png("cropped_cnt", img)
    return img


def imwrite_contours(name, gray, cnts, thick=2, padding=0, ignore=(sys.gettrace() is None)):
    if ignore:
        return
    gray = gray.copy()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(gray, (x+padding, y+padding), (x+w-padding, y+h-padding), (0, 0, 0), thick)
    imwrite_png(name, gray)


def biggest_children(cnts, hierarchy, parent_positon):
    # Look only to the children (startng with the biggest contours) to try to find a matricule
    n = hierarchy[0][parent_positon][2]  # first child index of the biggest contour
    scnts = []
    while n != -1:
        scnts.append(cnts[n])
        n = hierarchy[0][n][0]  # next index of the current contour
    return sorted(scnts, key=cv2.contourArea, reverse=True)


def find_digit_contours(gray, split_on_semi_column=True, min_box_before_split=0, max_cnts=None):
    # threshold the gray image, then apply a series of morphological
    # operations to cleanup the thresholded image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    imwrite_png("blurred", blurred)
    thresh = cv2.threshold(blurred, 200, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    imwrite_png("thresh", thresh)
    # # noise removal
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # imwrite_png("opening", opening)
    opening = thresh

    # finding contours in image
    cnts, h = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours((cnts, h))
    imwrite_contours("rgray", gray, cnts)

    # clean cnts
    scnts, dot = clean_and_sort_digit_contours(cnts, gray, split_on_semi_column, min_box_before_split, max_cnts)

    return scnts, dot, thresh


def clean_and_sort_digit_contours(cnts, gray=None, split_on_semi_column=True, min_box_before_split=0, max_cnts=None):
    # remove thin contours
    ccnts = []
    max_h = 0
    middle_y = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # remove thin contours, but not small ones (for example dots)
        if w < 10 ^ h < 10:
            continue
        if h > max_h:
            max_h = h
            middle_y = y + h/2
        ccnts.append(c)
    cnts = ccnts

    # remove anything that is too far from middle_y
    ccnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # remove if too far above or below
        if y + h < middle_y - max_h or y > middle_y + max_h:  # remove above or below
            continue
        ccnts.append(c)
    cnts = ccnts

    # sort contours by x
    scnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        scnts.append((x + w/2, w, c))
    scnts = sorted(scnts, key=lambda t: t[0])

    if gray is not None:
        imwrite_contours("rgray_all", gray, [c[-1] for c in scnts])

    # if split on last semi column: find two boxes that are similar
    if split_on_semi_column:
        prev_mx = -1
        prev_w = -1
        semi_column = -1
        i = 0
        for mx, w, c in scnts:
            if abs(mx - prev_mx) < 5 and abs(w - prev_w) < 5:
                semi_column = i
            prev_mx = mx
            prev_w = w
            i += 1
        if semi_column+1 < min_box_before_split:
            return [], 0
        if semi_column > -1:
            scnts = scnts[semi_column+1:]
    cnts = [c[-1] for c in scnts]

    if max_cnts and len(cnts) > max_cnts:
        return [], 0

    # keep centered contours
    # look for the middle line and remove anything above or below
    # and check for a dot
    dot = len(cnts)
    ccnts = []
    if gray is not None:
        gray = gray.copy()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if y+h < middle_y:  # remove above
            continue
        if y > middle_y:  # remove below
            if len(ccnts) < dot:  # store position of the first one, as it could be a dot
                dot = len(ccnts)
            continue
        ccnts.append(c)
    if gray is not None:
        imwrite_contours("rgray", gray, ccnts)

    return ccnts, dot


def extract_all_digits(cnts, gray, thresh, classifier, threshold=1e-2):
    all_digits = []
    for c in cnts:
        try:
            # creating a mask
            mask = np.zeros(gray.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)

            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)

            # Getting Region of interest
            roi = mask[max(0, y - 7):min(mask.shape[0], y + h + 7),
                       max(0, x - 7):min(mask.shape[1], x + w + 7)]
            imwrite_png("roi", roi)

            roi = make_square(roi)
            imwrite_png("roi2", roi)

            # predicting
            roi = roi / 255  # normalize
            roi = roi.reshape(1, 28, 28, 1).astype('float32')
            pproba = classifier.predict(roi)
            predict = [(p, i) for i, p in enumerate(pproba[0])]
            predict = sorted(predict, reverse=True)
            cumul = 0
            d = []
            for p, i in predict:
                d.append((p, i))
                cumul += p
                if cumul > 1 - threshold:
                    break
            all_digits.append((c, d))

        except Exception as e:
            print(e)

    return all_digits


def process_digits_combinations(all_digits, dot):
    # create all combinations
    combinations = [(0, [])]
    for (c, d) in all_digits:
        c2 = [(cumul + p, digits + [(c, i)]) for (p, i) in d for (cumul, digits) in combinations]
        combinations = c2
    combinations = sorted(combinations, reverse=True)
    # process all combinations: normalize probability and extract number
    numbers = []
    for p, digits in combinations:
        number = extract_number(digits, dot)
        if number:
            numbers.append((p / len(digits), number))
    return numbers


def extract_number(digits, dot, just_allowed_decimals=True):
    # create number
    number = ""
    decimals = ""
    for i, d in enumerate(digits):
        if i < dot:
            number = "%s%d" % (number, d[1])
        else:
            decimals = "%s%d" % (decimals, d[1])

    # check if decimals are allowed when enable
    if just_allowed_decimals and decimals and decimals not in allowed_decimals:
        print("Found decimals not allowed: %s is not within %s."
              % (decimals, ",".join(allowed_decimals)))
        return None

    return float("%s.%s" % (number, decimals))


def make_square(img, size=28, margin=.1):
    # size: size of the square
    # margin: margin in % of the square
    # get size
    h, w = img.shape
    # Create a black image
    s = int((1+margin)*max(w, h))
    square = np.zeros((s, s), np.uint8)
    y = int((s - h) / 2)
    x = int((s - w) / 2)
    square[y:y+h, x:x+w] = img
    return cv2.resize(square, (size, size))


def imwrite_png(name, img, ignore=(sys.gettrace() is None)):
    if ignore:
        return
    if img.shape[0] == 0 or img.shape[1] == 0:
        return
    os.mkdir('images')
    cv2.imwrite("images/%s.png" % name, img)
