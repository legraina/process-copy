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

# References:
# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

# import the necessary packages
import os
from colorama import Fore, Style
import re
import numpy as np, cv2, imutils
import pandas as pd
from pdf2image import convert_from_path
from keras.models import load_model


def grade_all(path, grades_csv, box, compare=False):
    # load csv
    grades_df = pd.read_csv(grades_csv, index_col='Matricule')

    # train()
    # gray = cv2.cvtColor(cv2.imread("gray.png"), cv2.COLOR_BGR2GRAY)
    # test(gray)

    # loading our CNN model
    classifier = load_model('digit_recognizer.h5')

    # f = 'Djoko Lionel_15482718_2071596_assignsubmission_file_Devoir6_MTH1002_Automne2021_Gr 2.pdf'
    # f2 = 'Bibombe Nathan_15482710_2118644_assignsubmission_file_Bibombe_Nathan_2118644_Devoir6_MTH1102_AUTOMNE2021_Gr02.pdf'
    # file = os.path.join(path, f)
    # numbers = grade(file, box, classifier)

    # grade files
    tp = 0
    fp = 0
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
            try:
                numbers = grade(file, box, classifier)
                print("%s: %.2f" % (f, numbers[-1]))
                if grades_df.at[m, 'Note'] == numbers[-1]:
                    tp += 1
                else:
                    fp += 1
                grades_df.at[m, 'Note'] = numbers[-1]
            except ValueError as e:
                print(e)
                print(Fore.GREEN + "%s: No valid grade" % f + Style.RESET_ALL)
                fn += 1
                grades_df.at[m, 'Note'] = -1
            n += 1
    # store grades
    if compare:
        print("Accuracy: %.3f, Precision: %.3f" % (tp / n, tp / (tp+fp)))
    else:
        grades_df.to_csv(grades_csv)


def grade(fpdf, box, classifier=None):
    # fpdf: path to pdf file to grade
    gray = gray_front_page_image(fpdf)

    cropped = fetch_box(gray, box)
    boxes = find_boxes(cropped)
    all_numbers = []
    for b in boxes:
        (x, y, w, h) = cv2.boundingRect(b)
        box_img = cropped[y + 5:y + h - 5, x + 5:x + w - 5]
        all_numbers.append(test(box_img.copy(), classifier))

    if len(all_numbers) == 0:
        raise ValueError("No numbers have been found")

    # find all combination that works
    combinations = [(0, [])]
    for numbers in all_numbers:
        c2 = [(c+p, l+[i]) for p, i in numbers for c, l in combinations]
        combinations = c2
    combinations = sorted(combinations, reverse=True)
    for p, numbers in combinations:
        # if only one number, return it
        if len(numbers) <= 1:
            return numbers
        # check the sum
        if sum(numbers) != 2*numbers[-1]:
            continue
        return numbers

    # for p, numbers in combinations:
    #     print("proba=%.3f, number: %s" % (p, str(numbers)))
    raise ValueError("Either the total is wrong or some numbers have not been correctly recognized.")


def gray_front_page_image(fpdf, dpi=300):
    # fpdf: path to pdf file to grade
    images = convert_from_path(fpdf, dpi=dpi, last_page=1, first_page=0)
    fimg = 'front_page.jpg'
    images[0].save(fimg, 'JPEG')
    img = cv2.imread(fimg)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fetch_box(img, box):
    # box = (x1, x2, y1, y2) in %
    x = [int(box[0] * img.shape[1]), int(box[1] * img.shape[1]),
         int(box[2] * img.shape[0]), int(box[3] * img.shape[0])]
    cropped = img[x[2]:x[3], x[0]:x[1]]  # ys and then xs
    imwrite_png("cropped", cropped)
    return cropped


def find_boxes(cropped):
    """ Find the lines on the image. """
    # Computing the edge map via the Canny edge detector
    edged = cv2.Canny(cropped, 50, 200, 255)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)
    for ll in lines:
        l = ll[0]
        cv2.line(edged, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 30)
    # util.drawLines(edged, lines, thickness=10)
    imwrite_png("edged", edged)

    """ Find boxes on the image. """
    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the contours:
    # 1. remove the first one that contours the table
    # 2. then take the biggest ones that are aligned
    # 3. Then stop
    cropped2 = cropped.copy()
    boxes = []
    ref = None
    horizontal = None
    for c in cnts[1:]:
        (x, y, w, h) = cv2.boundingRect(c)
        # set the reference box
        if ref is None:
            ref = (x, y, w, h)
        # box too small
        elif w * h < .3 * ref[2] * ref[3]:
            break
        # for horizontal alignment
        elif abs(ref[1] - y) <= 20:
            if horizontal is None:
                horizontal = True
            # break the alignment
            elif not horizontal:
                continue
            # break the box size
            if abs(ref[3] - h) >= 20:
                continue
        # for vertical alignment
        elif abs(ref[0] - x) <= 20:
            if horizontal is None:
                horizontal = False
            # break the alignment
            elif horizontal:
                continue
            # break the box size
            elif abs(ref[2] - w) >= 20:
                continue
        # break alignment
        else:
            continue
        boxes.append(c)
        cv2.rectangle(cropped2, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 0), 20)
    imwrite_png("cropped_boxes", cropped2)

    # sort boxes according to alignment
    sboxes = []
    for b in boxes:
        (x, y, w, h) = cv2.boundingRect(b)
        sboxes.append((x if horizontal else y, b))
    sboxes = sorted(sboxes)
    return [b for (p, b) in sboxes]


def test(gray_img, classifier=None):
    if classifier is None:
        classifier = load_model('digit_recognizer.h5')

    # image copy
    gray = gray_img.copy()

    # threshold the gray image, then apply a series of morphological
    # operations to cleanup the thresholded image
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    imwrite_png("blurred", blurred)
    thresh = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    imwrite_png("thresh", thresh)

    # finding countours in image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # clean cnts
    scnts, dot = clean_and_sort(cnts, gray)

    # extract digits
    all_digits = extract_all_digits(scnts, gray, thresh, classifier)
    # create all combinations
    combinations = [(0, [])]
    for (c, d) in all_digits:
        c2 = [(cumul+p, digits+[(c, i)]) for (p, i) in d for (cumul, digits) in combinations]
        combinations = c2
    combinations = sorted(combinations, reverse=True)
    # process all combinations
    numbers = []
    for p, digits in combinations:
        numbers.append((p, extract_number(digits, dot)))

    return numbers


def clean_and_sort(cnts, gray=None):
    # remove small contours
    ccnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        ccnts.append(c)

    # sort contours by x
    # remove non center numbers
    # look for the middle line and remove anything above or below
    max_h = 0
    middle_y = 0
    scnts = []
    for c in ccnts:
        (x, y, w, h) = cv2.boundingRect(c)
        scnts.append((x + w/2, c))
        if h > max_h:
            max_h = h
            middle_y = y + h/2
    scnts = sorted(scnts)

    # keep centered contours
    # and check for a dot
    cnts = []
    dot = -1
    if gray is not None:
        gray = gray.copy()
    for mx, c in scnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if y+h < middle_y:  # remove above
            continue
        if y > middle_y:  # remove below
            if dot == -1:  # store position of the first one, as it could be a dot
                dot = len(cnts)
            continue
        cnts.append(c)
        if gray is not None:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), 2)
    if gray is not None:
        imwrite_png("rgray", gray)

    return cnts, dot if dot < len(cnts) else -1


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
            pproba = classifier.predict_proba(roi)
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


def extract_number(digits, dot):
    # create number
    number = ""
    for i, d in enumerate(digits):
        if i == dot:
            number = "%s." % number
        number = "%s%d" % (number, d[1])
    return float(number)


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


def train(mix_datasets=True):
    # train dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(mnist=mix_datasets)
    if mix_datasets:
        train_for(x_train, y_train, x_test, y_test, "mix")
    else:
        train_for(x_train, y_train, x_test, y_test, "dataset")
        # train mnist next
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        train_for(x_train, y_train, x_test, y_test, "mnist")


def train_for(x_train, y_train, x_test, y_test, name, epochs=5):
    # reshape to be [samples][width][height][channels]
    # normalize inputs from 0-255 to 0-1
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # one hot encode outputs
    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # build the model
    num_classes = y_test.shape[1]
    model = baseline_model(num_classes)

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s CNN Error: %.2f%%" % (name, 100 - scores[1] * 100))

    model.save('digit_recognizer_%s.h5' % name)


def load_dataset(test_ratio=.2, mnist=True):
    dataset_file = 'dataset.npy'
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            x = np.load(f)
            y = np.load(f)
    else:
        x, y = create_dataset(dataset_file)

    if mnist:
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x, x_train, x_test))
        y = np.concatenate((y, y_train, y_test))

    # shuffle datasets
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    # return train set and test set
    ti = int(x.shape[0] * (1 - test_ratio))
    return (x[:ti], y[:ti]), (x[ti:], y[ti:])


def display_dataset(x, y, prefix, n=10):
    for i in range(n):
        j = np.random.randint(0, x.shape[0])
        imwrite_png('%s%d_%d' % (prefix, y[j], i), x[j])


def create_dataset(dataset_file):
    x = np.empty((0, 28, 28), dtype='uint8')
    y = np.array([], dtype='int')
    # load dataset (https://www.kaggle.com/jcprogjava/handwritten-digits-dataset-not-in-mnist)
    for d in range(10):
        p = 'dataset/%d' % d
        n = x.shape[0]
        for f in os.listdir(p):
            if f.endswith('png'):
                img = cv2.imread(os.path.join(p, f), cv2.IMREAD_UNCHANGED)  # for transparency
                gray = 255 - np.sum(img, axis=2)  # convert transparancy channel to B&W
                x = np.vstack((x, gray.reshape(1, 28, 28)))
        y = np.concatenate((y, np.full((x.shape[0] - n), d)))
        display_dataset(x, y, '%d_' % d)

    with open(dataset_file, 'wb') as f:
        np.save(f, x)
        np.save(f, y)

    return x, y


def baseline_model(num_classes):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D

    # create model
    model = Sequential()

    # 1
    # model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))

    # 2: https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def imwrite_png(name, img):
    if img.shape[0] == 0 or img.shape[1] == 0:
        return
    cv2.imwrite("images/%s.png" % name, img)
