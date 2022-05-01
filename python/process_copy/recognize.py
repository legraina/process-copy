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
import unidecode
from colorama import Fore, Style
import re
import numpy as np, cv2, imutils
import pandas as pd
from keras.models import load_model
from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime
from process_copy.config import re_mat
from process_copy.config import MoodleFields as MF
from process_copy.mcc import get_name, load_csv


allowed_decimals = ['0', '25', '5', '75']
corrected_decimals = ['5', '75']  # for length 1, use first one, lenght 2, use second one ...
len_mat = 7
RED = (225,6,0)
GREEN = (0,154,23)
ORANGE = (255,127,0)
BLACK=(0,0,0)

ph = 0
pw = 0
half_dpi = 0
quarter_dpi = 0
one_height_dpi = 0


def refresh(dpi=300):
    global ph, pw, half_dpi, quarter_dpi, one_height_dpi
    ph = int(11 * dpi)
    pw = int(8.5 * dpi)
    half_dpi = int(dpi / 2)
    quarter_dpi = int(dpi / 4)
    one_height_dpi = int(dpi / 8)


refresh()


def find_matricules(paths, box, grades_csv=[], dpi=300, shape=(8.5, 11)):
    shape = (int(dpi * shape[0]), int(dpi * shape[1]))

    # loading our CNN model
    classifier = load_model('digit_recognizer.h5')

    # load csv
    grades_dfs, grades_names = load_csv(grades_csv)
    root_dir = None

    # list files and directories
    matricules_data = {}
    duplicates = set()
    invalid = []
    for path in paths:
        r = os.path.dirname(path)
        if not root_dir:
            root_dir = r
        elif root_dir.count('/') > r.count('/'):
            root_dir = r

        for root, dirs, files in os.walk(path):
            for f in files:
                if not f.endswith('.pdf'):
                    continue
                file = os.path.join(root, f)
                if os.path.isfile(file):
                    grays = gray_images(file, shape=shape)
                    if grays is None:
                        print(Fore.RED + "%s: No valid pdf" % f + Style.RESET_ALL)
                        continue
                    mat, id_box, id_group = find_matricule(grays, box['front'], box['regular'], classifier, grades_dfs,
                                                           separate_box=box['separate_box'])
                    name = grades_dfs[id_group].at[mat, MF.name] if id_group is not None else mat
                    if name:
                        name = unidecode.unidecode(name)
                    if not mat:
                        print(Fore.RED + "No matricule found for %s" % f + Style.RESET_ALL)
                    else:
                        print("Matricule %s found for %s. Name: %s" % (mat, f, name))

                    m = mat if mat else "NA"
                    if m not in matricules_data:
                        matricules_data[m] = []
                        # if no valid matricule has been found
                        if m != "NA" and grades_dfs and id_group is None:
                            invalid.append(m)
                    elif m != "NA":
                        duplicates.add(m)
                    matricules_data[m].append((id_box, name, file))

    sumarries = []
    csvf = "Id,Matricule,NomComplet,File\n"

    def add_summary(mat, id_box, name, file, invalid=False, initial_index=1):
        i = len(sumarries)+initial_index
        l_csv = '%d,%s,%s,%s\n' % (i, mat if mat else '', name if name else '', file)
        sumarry = create_summary(id_box, name, None, None,
                                 "%d: %s" % (i, file.rsplit('/')[-1]), dpi,
                                 align_matricule_left=False, name_bottom=False, invalid=invalid)
        sumarries.append(sumarry)
        return l_csv

    print(Fore.RED)
    if 'NA' in matricules_data:
        for id_box, name, file in matricules_data['NA']:
            print("No matricule found for %s" % file)
            csvf += add_summary(None, id_box, None, file)
        matricules_data.pop('NA')

    for m in sorted(invalid):
        print("No valid matricule %s for:" % m)
        for id_box, name, file in matricules_data[m]:
            print("    " + file)
            csvf += add_summary(m, id_box, None, file, invalid=True)
        matricules_data.pop(m)

    for m in sorted(duplicates):
        print("Duplicate files found for matricule %s:" % m)
        for id_box, name, file in matricules_data[m]:
            print("    " + file)
            csvf += add_summary(m, id_box, name, file, invalid=True)
        matricules_data.pop(m)
    print(Style.RESET_ALL)

    for m in sorted(matricules_data):
        if len(matricules_data[m]) != 1:
            raise ValueError('The list should contain only one element associated to a given matricule (%s)' % m)
        id_box, name, file = matricules_data[m][0]
        csvf += add_summary(m, id_box, name, file)

    # save summary pdf and grades
    pages = create_whole_summary(sumarries)
    save_pages(pages, os.path.join(root_dir, "matricule_summary.pdf"))
    with open(os.path.join(root_dir, "matricules.csv"), 'w') as wf:
        wf.write(csvf)


def grade_all(paths, grades_csv, box, id_box=None, dpi=300, shape=(8.5,11)):
    shape = (int(dpi * shape[0]), int(dpi * shape[1]))

    # load csv
    grades_dfs, grades_names = load_csv(grades_csv)

    # load max grade if available
    max_grade = None
    for df in grades_dfs:
        for idx, row in df.iterrows():
            s = row[MF.max]
            if pd.isna(s):
                continue
            if isinstance(s, str):
                s = s.replace(',', '.')
                try:
                    s = float(s)
                except:
                    continue
            if max_grade is None or s < max_grade:
                max_grade = s

    # loading our CNN model
    classifier = load_model('digit_recognizer.h5')

    # grade files
    grades_data = []
    dt = get_date()
    trim = box['trim'] if 'trim' in box else None
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                if not f.endswith('.pdf'):
                    continue
                # search matricule
                m = re.search(re_mat, f)
                if not m:
                    print("Matricule wasn't found in "+f)
                    continue
                m = m.group()

                # try to recognize each grade and verify the total
                file = os.path.join(root, f)
                if os.path.isfile(file):
                    grays = gray_images(file, [0], straighten=False, shape=shape)
                    if grays is None:
                        print(Fore.RED + "%s: No valid pdf" % f + Style.RESET_ALL)
                        continue
                    gray = grays[0]
                    total_matched, numbers, grades = grade(gray, box['grade'],
                                                           classifier=classifier, trim=trim, max_grade=max_grade)
                    i, name = get_name(m, grades_dfs)
                    if i < 0:
                        print(Fore.RED + "%s: Matricule (%s) not found in csv files" % (f, m) + Style.RESET_ALL)
                    # fill moodle csv file
                    if numbers:
                        if pd.isna(grades_dfs[i].at[m, MF.grade]):
                            print("%s: %.2f" % (f, numbers[-1]))
                            grades_dfs[i].at[m, MF.grade] = numbers[-1]
                            grades_dfs[i].at[m, MF.mdate] = dt
                        elif grades_dfs[i].at[m, MF.grade] != numbers[-1]:
                            print(Fore.RED + "%s: there is already a grade (%.2f) different of %.2f" %
                                  (f, grades_dfs[i].at[m, MF.grade], numbers[-1]) + Style.RESET_ALL)
                        else:
                            print("%s: found same grade %.2f" % (f, numbers[-1]))
                    else:
                        print(Fore.GREEN + "%s: No valid grade" % f + Style.RESET_ALL)
                        grades_dfs[i].at[m, MF.mdate] = dt

                    # Display in the summary the identity box if provided
                    id_img = None
                    if id_box:
                        # find the id box
                        cropped = fetch_box(gray, id_box['front'])
                        cnts = cv2.findContours(find_edges(cropped, thick=0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        imwrite_contours("id_gray", cropped, cnts, thick=5)
                        # Find the biggest contour for the front box
                        pos, biggest_c = max(enumerate(cnts), key=lambda cnt: cv2.contourArea(cnt[1]))
                        id_img = get_image_from_contour(cropped, biggest_c)

                    grades_data.append((m, i, f, grades, numbers, total_matched, id_img))

    # check the number of files that have benn dropped on moodle if any
    n = 0
    for df in grades_dfs:
        for idx, row in df.iterrows():
            s = row[MF.status]
            if pd.isna(s):
                continue
            if s.startswith(MF.status_start_filter):
                n += 1
    if n > 0 and n != len(grades_data):
        print(Fore.RED + "%d copies have been uploaded on moodle, but %d have been graded" % (n, len(grades_data))
              + Style.RESET_ALL)

    # add summarry
    sumarries = [[] for f in grades_csv]

    def add_summary(file, grades, mat, numbers, total_matched, id_group, id_img=None, initial_index=2):
        lsum = sumarries[id_group]
        # rename file
        name = "%d: %s" % (len(lsum)+initial_index, file)  # recover id box if provided
        if id_img is not None:
            sumarry = create_summary2(id_img, grades, mat, numbers, total_matched, name, dpi)
        else:
            sumarry = create_summary(grades, mat, numbers, total_matched, name, dpi)
        lsum.append(sumarry)

    grades_data = sorted(grades_data)
    for mat, id_group, file, grades, numbers, total_matched, id_img in grades_data:
        add_summary(file, grades, mat, numbers, total_matched, id_group, id_img)

    # write summary
    for i, f in enumerate(grades_csv):
        pages = create_whole_summary(sumarries[i])
        gname = f.split('.')[0]
        save_pages(pages, gname + "_summary.pdf")
        # store grades
        df = grades_dfs[i]
        # sort by status (Remis in first) then matricules (index)
        status = np.array([not pd.isna(v) and not v.startswith('Remis') for v in df.Statut.values])
        sorted_indexes = np.lexsort((df.index.values, status))
        sdf = df.iloc[sorted_indexes]
        sdf.to_csv(f)


def compare_all(paths, grades_csv, box, dpi=300, shape=(8.5, 11)):
    shape = (int(dpi * shape[0]), int(dpi * shape[1]))

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
    for path in paths:
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
                grays = gray_images(file, [0], straighten=False, shape=shape)
                if grays is None:
                    print(Fore.RED + "%s: No valid pdf" % f + Style.RESET_ALL)
                    continue
                gray = grays[0]
                total_matched, numbers = grade(gray, box, classifier)
                if numbers:
                    print("%s: %.2f" % (f, numbers[-1]))
                    if grades_df.at[m, MF.grade] == numbers[-1]:
                        if total_matched:
                            tp += 1
                        else:
                            tpp += 1
                    elif total_matched:
                        fp += 1
                    else:
                        fpp += 1
                    grades_df.at[m, MF.grade] = numbers[-1]
                else:
                    print(Fore.GREEN + "%s: No valid grade" % f + Style.RESET_ALL)
                    fn += 1
                    grades_df.at[m, MF.grade] = -1
                n += 1
    # store grades
    print("Accuracy: %.3f (%.3f, %.3f), Precision: %.3f (%.3f, %.3f)"
          % ((tp+tpp) / n, tp / n, tpp / n, (tp+tpp) / (tp+tpp+fp+fpp), tp / (tp+fp), tpp / (tpp+fpp)))


def find_matricule(grays, front_box, regular_box, classifier, grades_dfs=[], separate_box=True):
    possible_digits = [{} for i in range(len_mat)]

    def find_digits(gray_box, split=False):
        try:
            # find contours of the numbers.
            # If separate_box, each number of the matricule is in its separate box
            # return a sorted list of the relevant digits' contours and the dot position (and the threshold image used)
            # 10 = len("Matricule:")
            cnts, dot, thresh = find_digit_contours(gray_box, max_cnts=len_mat,
                                                    split_on_semi_column=split, min_box_before_split=6)
            # check length
            if len(cnts) != len_mat:
                return False

            all_digits = []
            # if each number is in a separate box, extract it individually
            if separate_box:
                for c in cnts:
                    digit_box = get_image_from_contour(gray_box, c, border=7)
                    dcnts, dot, dthresh = find_digit_contours(digit_box)
                    # check if only one digit has been found in the box
                    if len(dcnts) == 1:
                        d = extract_digit(dcnts[0], digit_box, dthresh, classifier)
                    # if too many contours, just remove some pixels on the border of the image
                    elif len(dcnts) > 1:
                        d = extract_digit(c, gray_box, thresh, classifier, border=-7)
                    # if no contour at all, it means that at least one of the box is empty
                    # -> throw this results
                    else:
                        all_digits = []
                        break
                    all_digits.append(d)
            # otherwise, extract all digits at once
            else:
                all_digits = extract_all_digits(cnts, gray_box, thresh, classifier)
                all_digits = [d for c, d in all_digits]
        except cv2.error as e:
            print(e)
            print('Got an error while finding digits.')
            return False

        # check length
        if len(all_digits) != len_mat:
            return True

        # store values
        for i, digits in enumerate(all_digits):
            distri = possible_digits[i]
            for p, d in digits:
                if d in distri:
                    distri[d] += p
                else:
                    distri[d] = p
        return True

    # find the id box
    cropped = fetch_box(grays[0], front_box)
    cnts, hierarchy = cv2.findContours(find_edges(cropped, thick=0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours((cnts, hierarchy))
    imwrite_contours("rgray", cropped, cnts, thick=5)
    # Find the biggest contour for the front box
    pos, biggest_c = max(enumerate(cnts), key=lambda cnt: cv2.contourArea(cnt[1]))
    id_box = get_image_from_contour(cropped, biggest_c)
    for cnt in biggest_children(cnts, hierarchy, pos):
        cnt_cropped = get_image_from_contour(cropped, cnt)
        if find_digits(cnt_cropped, True):
            break

    # try to find a matricule on the next page
    for gray in grays[1:]:
        cropped = fetch_box(gray, regular_box)
        # mgray = find_edges(cropped, thick=3, line_on_original=True, max_gap=5, min_lenth=150)
        find_digits(cropped, True)

    # build matricules and sort them by probabilities
    matricules = [(0, '')]
    for distri in possible_digits:
        matricules = [(c+p, '%s%d' % (m, d)) for c, m in matricules for d, p in distri.items()]
    smats = sorted(matricules, reverse=True)

    # find the most probable matricule that exists
    if grades_dfs:
        for p, mat in smats:
            i, name = get_name(mat, grades_dfs)
            if i >= 0:
                return mat, id_box, i
    # find the most valid and probable one if no csv (or not valid) to check the matricule
    for p, mat in smats:
        if re.match(re_mat, mat):
            return mat, id_box, None

    return None, id_box, None


def grade(gray, box, classifier=None, add_border=False, trim=None, max_grade=None):
    cropped = fetch_box(gray, box)
    boxes = find_grade_boxes(cropped, add_border, thick=0)
    all_numbers = []
    for i, b in enumerate(boxes):
        (x, y, w, h) = cv2.boundingRect(b)
        if h <= 10 or w <= 10:
            print("An invalid box number has been found (too small or too thin)")
            return False, None, cropped
        box_img = cropped[y + 5:y + h - 5, x + 5:x + w - 5]
        # check if need to trim
        n_trim = None
        if trim:
            for (j, n) in trim:
                if i == j or i == len(boxes) + j:
                    n_trim = n
                    break
            # trim everything
            if n_trim < 0:
                continue
        all_numbers.append(test(box_img.copy(), classifier, trim=n_trim))

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
    # Try first the ones with the highest probability and below the max grade if available
    combinations = sorted(combinations, reverse=True)
    for p, numbers in combinations:
        # if only one number, return it
        if len(numbers) <= 1:
            return True, numbers, cropped
        # check the sum
        total = sum(numbers[:-1])
        if max_grade is not None and numbers[-1] > max_grade:
            continue
        if total != numbers[-1]:
            continue
        return True, numbers, cropped

    # Has not been able to check the total -> give the best prediction
    expected_numbers = [n[0] for n in all_numbers[:-1]]
    p, total = (sum(n[0] for n in expected_numbers) / len(expected_numbers),
                sum(n[1] for n in expected_numbers))
    pt, nt = all_numbers[-1][0]
    # keep the first numbers, but choose the most probable feasible total (either the sum or the one read)
    # when nt is 0, always choose the total
    adjusted_total = total
    if pt >= p and nt > 0:
        adjusted_total = nt
    if max_grade is not None:
        if nt > max_grade:
            adjusted_total = total
        if total > max_grade and nt > 0:
            adjusted_total = nt
        # if both are greater then max grade, both are false !
    return False, [n for p, n in expected_numbers[:-1]] + [adjusted_total], cropped


def test(gray_img, classifier=None, trim=None):
    if classifier is None:
        classifier = load_model('digit_recognizer.h5')

    # image copy
    gray = gray_img.copy()
    imwrite_png("gray", gray)

    # find contours of the numbers as well as the dot number position
    # return a sorted list of the relevant digits' contours and the dot position (and the threshold image used)
    cnts, dot, thresh = find_digit_contours(gray, trim=trim)

    # if found no digits contours, return 0
    if not cnts:
        return [(1.0, 0)]

    # extract digits
    all_digits = extract_all_digits(cnts, gray, thresh, classifier)

    if not all_digits:
        print("No valid number has been found")
        return []

    # process all possible digits combinations
    return process_digits_combinations(all_digits, dot)


def clean_and_sort_digit_contours(cnts, gray=None, split_on_semi_column=True,
                                  min_box_before_split=0, max_cnts=None, trim=None):
    # remove thin contours
    ccnts = []
    max_h = 0
    middle_y = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # remove thin contours (dot should not been removed)
        if w < 5 or h < 5:
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
        scnts.append((x + w/2, w, h, c))
    scnts = sorted(scnts, key=lambda t: t[0])

    # trim if needed
    if trim and len(scnts) > trim:
        scnts = scnts[:-trim]

    if gray is not None:
        imwrite_contours("rgray_all", gray, [c[-1] for c in scnts])

    # if split on last semi column: find two boxes that are similar
    if split_on_semi_column:
        prev_mx = -1
        prev_w = -1
        prev_h = -1
        semi_column = -1
        i = 0
        for mx, w, h, c in scnts:
            if abs(mx - prev_mx) < 5 and abs(w - prev_w) < 5 and abs(h - prev_h) < 5:
                semi_column = i
            prev_mx = mx
            prev_w = w
            prev_h = h
            i += 1
        if semi_column+1 < min_box_before_split:
            return [], 0
        if semi_column > -1:
            scnts = scnts[semi_column+1:]
    cnts = [c[-1] for c in scnts]

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
        imwrite_contours("dgray_cnt", gray, ccnts)

    if max_cnts and len(ccnts) > max_cnts:
        return [], 0

    return ccnts, dot


def extract_all_digits(cnts, gray, thresh, classifier, threshold=1e-2, border=7):
    all_digits = []
    for c in cnts:
        try:
            d = extract_digit(c, gray, thresh, classifier, threshold, border)
            all_digits.append((c, d))
        except Exception as e:
            print(e)
    return all_digits


def extract_digit(cnt, gray, thresh, classifier, threshold=1e-2, border=7):
    # creating a mask
    mask = np.zeros(gray.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(cnt)

    hull = cv2.convexHull(cnt)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Getting Region of interest
    roi = mask[max(0, y - border):min(mask.shape[0], y + h + border),
               max(0, x - border):min(mask.shape[1], x + w + border)]
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
    return d


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
        if number is not None:
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
        # try to correct decimals
        l = len(decimals)
        if l <= len(corrected_decimals):
            decimals = corrected_decimals[l-1]
        else:
            print("Found decimals not allowed: %s is not within %s."
                  % (decimals, ",".join(allowed_decimals)))
            return None

    return float("%s.%s" % (number, decimals))


def find_edges(cropped, thick=5, min_lenth=80, max_gap=15, angle_resolution=np.pi/2, line_on_original=False):
    """ Find the lines on the image. """
    # Computing the edge map via the Canny edge detector
    edged = cv2.Canny(cropped, 50, 200, 255)
    imwrite_png("canny", edged)
    edged = cv2.dilate(edged, kernel=np.ones((3, 3), dtype='uint8'))
    imwrite_png("dilated", edged)

    if thick:
        lines = cv2.HoughLinesP(edged, 1, angle_resolution, 50, minLineLength=min_lenth, maxLineGap=max_gap)
        if line_on_original:
            edged = cropped.copy()
        if lines is not None:
            for l in lines:
                cv2.line(edged, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255, 255, 255), thick)
            imwrite_png("edged", edged)

        # erode the image to keep only the big lines
        if not line_on_original:
            edged = cv2.erode(edged, kernel=np.ones((thick, thick), dtype='uint8'))
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
    imwrite_contours("cropped_all_boxes", cropped2, cnts, thick=thick+1)

    # check if any contour
    if not cnts:
        return []

    # keep only the children of the biggest contour
    pos, c = max(enumerate(cnts), key=lambda cnt: cv2.contourArea(cnt[1]))
    imwrite_png("main_cropped", get_image_from_contour(cropped, c))
    ccnts = biggest_children(cnts, hierarchy, pos)

    # Retrieve
    # loop over the contours to take the ones that are aligned with the biggest one and are big enough
    cropped2 = cropped.copy()
    boxes = []
    ref = None
    horizontal = None
    imwrite_contours("cropped_boxes", cropped2, ccnts, thick=thick+1)
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
        if boxes:
            (x, y, w, h) = cv2.boundingRect(boxes[-1])
            if x + y + h + w >= cropped.shape[0] + cropped.shape[1] - 4*thick - 5:
                boxes = boxes[:-1]
    # add any missing box
    if boxes:
        prev = None
        m_size = max(4 * thick, 20)
        boxes2 = []
        for b in boxes:
            (x, y, w, h) = cv2.boundingRect(b)
            if prev is not None:
                if horizontal:
                    # add a contour
                    if x - prev > m_size:
                        boxes2.append(np.array([[prev+thick, y-thick], [x-thick, y+h+thick]]))
                elif y - prev > m_size:  # add a contour
                    boxes2.append(np.array([[x-thick, prev+thick], [x+w+thick, y-thick]]))
            boxes2.append(b)
            prev = x+w if horizontal else y+h
        boxes = boxes2
    imwrite_contours("cropped_boxes", cropped2, boxes, thick=2*(thick+1), padding=-thick-1)
    return boxes


def gray_images(fpdf, pages=None, dpi=300, straighten=True, shape=None):
    # shape: width, height
    # fpdf: path to pdf file to grade
    images = []
    if pages is None:
        try:
            images = convert_from_path(fpdf, dpi=dpi)
        except Image.DecompressionBombError as e:
            print("Decompression issue for %s." % fpdf)
            return None
    else:
        for p in pages:
            try:
                images += convert_from_path(fpdf, dpi=dpi, last_page=p+1, first_page=p)
            except Image.DecompressionBombError as e:
                print("Decompression issue on page %d for %s." % (p, fpdf))
                return None
    gray_images = []
    for i, img in enumerate(images):
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        if straighten:
            try:
                gray = imstraighten(gray)
            except ValueError as e:
                print("For %s, page %d: %s" % (fpdf, i, str(e)))
        if shape:
            if abs(gray.shape[0] - shape[1]) > .1 * shape[1] \
                    or abs(gray.shape[1] - shape[0]) > .1 * shape[0]:
                print("Resizing %s, wrong format: %.1f by %.1f in." % (fpdf, gray.shape[0]/dpi, gray.shape[1]/dpi))
                gray = cv2.resize(gray, shape, interpolation=cv2.INTER_LINEAR)
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
    thresh = cv2.threshold(ngray, 10, 255, cv2.THRESH_BINARY)[1]
    imwrite_png("thresh", thresh)

    # # grab the (x, y) coordinates of all pixel values that
    # # are greater than zero, then use these coordinates to
    # # compute a rotated bounding box that contains all
    # # coordinates
    # coords = np.column_stack(np.where(thresh > 0))

    # get rid of thinner lines
    dilated = cv2.dilate(thresh, np.ones((5, 5), np.uint8))
    imwrite_png("dilated", dilated)
    # find lines
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, 50, minLineLength=half_dpi)
    if lines is None:
        return gray
    # find longuest lines -> should be horizontal or vertical
    max_dist = 0
    max_line = None
    for l in lines:
        d = np.square(l[0][2] - l[0][0]) + np.square(l[0][3] - l[0][1])
        if d > max_dist:
            max_dist = d
            max_line = l[0]
    coords = np.array([max_line[0:2], max_line[2:4]])
    center, dim, angle = cv2.minAreaRect(coords)
    mangle = (180 + angle) % 90
    if mangle > 45:
        mangle = 90 - mangle
    if abs(mangle) > 10:
        raise ValueError("Current page is too skewed (angle found: %.2f)." % angle)
    # rotate the image to deskew it
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    # divide angle by 2 in case of error as we are changing the center and we are just using small angles
    M = cv2.getRotationMatrix2D(center, mangle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    imwrite_png("rotated", rotated)
    return rotated


def get_image_from_contour(img, cnt, border=0):
    (x, y, w, h) = cv2.boundingRect(cnt)
    img = img[y+border:y+h-border, x+border:x+w-border]
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


def find_digit_contours(gray, split_on_semi_column=True, min_box_before_split=0, max_cnts=None, trim=None):
    thresh = get_clean_thresh(gray)

    # finding contours in image
    cnts, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours((cnts, h))
    imwrite_contours("rgray_cnt", gray, cnts)

    # clean cnts
    scnts, dot = clean_and_sort_digit_contours(
        cnts, gray, split_on_semi_column, min_box_before_split, max_cnts, trim=trim)

    return scnts, dot, thresh


def get_clean_thresh(gray):
    # threshold the gray image, then apply a series of morphological
    # operations to cleanup the thresholded image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    imwrite_png("blurred", blurred)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)[1]  # | cv2.THRESH_OTSU
    imwrite_png("thresh", thresh)
    return thresh


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
    if not os.path.exists('images'):
        os.mkdir('images')
    cv2.imwrite("images/%s.png" % name, img)


def get_blank_page(h=ph, w=pw, dim=None):
    if dim:
        return np.full((h, w, dim), 255, np.uint8)
    else:
        return np.full((h, w), 255, np.uint8)


def create_summary2(id_box, grades, mat, numbers, total_matched, name, dpi,
                   align_matricule_left=True, name_bottom=True, invalid=False):
    w = id_box.shape[1] + grades.shape[1] + dpi
    h = max(id_box.shape[0] + dpi, grades.shape[0])

    summary = get_blank_page(h, w)
    # add id
    summary[0:id_box.shape[0], 0:id_box.shape[1]] = id_box
    # add grades
    summary[0:grades.shape[0], id_box.shape[1] + dpi:w] = grades
    # write matricule and grade in color
    color_summary = cv2.cvtColor(summary, cv2.COLOR_GRAY2RGB)
    pos = int(2.5 * dpi) if align_matricule_left else id_box.shape[1] - int(4 * dpi)
    cv2.putText(color_summary, mat if mat else 'N/A',
                (pos, id_box.shape[0] + half_dpi),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN if mat or not invalid else RED, 5)
    if total_matched is not None:
        cv2.putText(color_summary, str(numbers[-1]) if numbers else 'N/A',
                    (id_box.shape[1] + half_dpi, h - half_dpi),  # position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN if total_matched else RED, 5)
    pos = h - one_height_dpi if name_bottom else one_height_dpi
    cv2.putText(color_summary, name, (quarter_dpi, pos),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)
    imwrite_png('summary', color_summary)
    return color_summary


def create_summary(grades, mat, numbers, total_matched, name, dpi,
                   align_matricule_left=True, name_bottom=True, invalid=False):
    # put everything in an image
    w = grades.shape[1]
    h = grades.shape[0]

    summary = get_blank_page(h, w)
    # add grades
    summary[0:h, 0:w] = grades
    # write matricule and grade in color
    color_summary = cv2.cvtColor(summary, cv2.COLOR_GRAY2RGB)
    pos = (one_height_dpi, h - half_dpi) if align_matricule_left else (grades.shape[1] - int(2.5 * dpi), half_dpi)
    cv2.putText(color_summary, str(mat) if mat else 'N/A',
                pos,  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN if mat and not invalid else RED, 5)
    if total_matched is not None:
        cv2.putText(color_summary, str(numbers[-1]) if numbers else 'N/A',
                    (w - dpi, h - half_dpi),  # position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN if total_matched else RED, 5)
    pos = h - one_height_dpi if name_bottom else one_height_dpi
    cv2.putText(color_summary, name, (one_height_dpi, pos),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)
    imwrite_png('summary', color_summary)
    return color_summary


def create_whole_summary(sumarries):
    # retrieve max width and height
    max_h = 0
    max_w = 0
    for s in sumarries:
        if s.shape[1] > max_w:
            max_w = s.shape[1]
        if s.shape[0] > max_h:
            max_h = s.shape[0]

    # create summary pages
    hmargin = int(pw - max_w) // 2  # horizontal margin
    f = 1
    if hmargin < half_dpi:
        hmargin = half_dpi
        w = pw - 2*half_dpi
        f = w / max_w
        max_w = w
        max_h = int(max_h * f)

    imgh = max_h + 15
    n_s = ph // imgh  # number of pictures by page
    vmargin = int(ph - n_s * imgh) // 2

    pages = []
    page = get_blank_page(dim=3)
    y = vmargin
    # put summaries on pages
    for s in sumarries:
        if f < 1:
            s = cv2.resize(s, None, fx=f, fy=f)
        # new page if needed
        if y + s.shape[0] > ph - vmargin:
            # store current page
            imwrite_png("page", page)
            pages.append(page)
            # create new one
            page = get_blank_page(dim=3)
            y = vmargin
        # add summarry
        page[y:y + s.shape[0], hmargin:hmargin + s.shape[1]] = s
        y += s.shape[0] + 5  # update cursor
        page[y:y + 2, :] = BLACK
        y += 7
    # store current page
    imwrite_png("page", page)
    pages.append(page)

    return pages


def save_pages(pages, fname):
    images = [Image.fromarray(p) for p in pages]
    images[0].save(fname, save_all=True, append_images=images[1:])


def get_date():
    # datetime object containing current date and time
    now = datetime.now()

    # dimanche 23 janvier 2022, 23:42
    return now.strftime("%A %d %B %Y, %H:%M")
