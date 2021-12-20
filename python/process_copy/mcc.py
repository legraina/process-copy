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

import os
import shutil
import glob
import zipfile


def copy_file(file, folder):
    shutil.copy(file, folder)


def copy_files(path, mpath=None):
    files = os.listdir(path)
    moodle_folders = os.listdir(mpath)
    n = 0
    for f in files:
        file = os.path.join(path, f)
        if os.path.isfile(file):
            try:
                matricule = f.split('_')[2]
            except IndexError:
                continue
            # find moodle folder
            folder = next(fd for fd in moodle_folders if matricule in fd)
            # copy file
            folder = os.path.join(mpath, folder)
            folder_files = glob.glob(folder+"/*")
            for f2 in folder_files:
                os.remove(f2)
            copy_file(file, os.path.join(mpath, folder))
            n += 1
    print('%d files has been moved and copied to %s.' % (n, mpath))


def zipdirrec(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            pfile = os.path.join(root, file)
            ziph.write(pfile, os.path.relpath(pfile, os.path.join(path, '..')))
        for dir in dirs:
            zipdirrec(os.path.join(path, dir), ziph)


def zipdirbatch(path, archive='moodle', batch=None):
    # ziph is zipfile handle
    i = 0
    j = 0
    narchive = archive
    ziph = zipfile.ZipFile(narchive+'.zip', 'w', zipfile.ZIP_DEFLATED)
    print("Compressing ", end="", flush=True)
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            zipdirrec(os.path.join(path, dir), ziph)
            i = i + 1
            if batch and i % batch == 0:
                print('\nArchive %s.zip created.' % narchive)
                j = j + 1
                narchive = "%s%d" % (archive, j)
                ziph = zipfile.ZipFile(narchive + '.zip', 'w', zipfile.ZIP_DEFLATED)
                print("Compressing ", end="", flush=True)
            else:
                print(".", end="" if i % 65 else "\nCompressing ", flush=True)
    print('\nArchive %s.zip created.' % narchive)
