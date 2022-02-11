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
import re
import pandas as pd

from process_copy.config import re_mat


MB = 2**20


def copy_file(file, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy(file, folder)


def copy_files(path, mpath=None, grades_csv=None):
    files = os.listdir(path)
    grades_df = pd.read_csv(grades_csv, index_col='Matricule') if grades_csv is not None else None
    moodle_folders = os.listdir(mpath) if grades_csv is None else None
    n = 0
    for f in files:
        file = os.path.join(path, f)
        if os.path.isfile(file):
            try:
                # search matricule
                m = re.search(re_mat, f)
                if not m:
                    print("Matricule wasn't found in " + f)
                    continue
                mat = m.group()
            except IndexError:
                continue
            except StopIteration as e:
                if f.endswith('.pdf'):
                    print("Matricule wasn't found in " + f)
                continue
            # find moodle folder
            # rebuild moodle folder name: "Nom complet_Identifiant_Matricule_assignsubmission_file_"
            if grades_df is not None:
                participant = grades_df.at[mat, 'Identifiant']
                m = re.search('\\d+', participant)
                if not m:
                    print("Moodle participant id not found in " + participant)
                    continue
                m_id = m.group()
                folder = "%s_%s_%s_assignsubmission_file_" % (grades_df.at[mat, 'Nom complet'], m_id, mat)
            else:
                folder = next(fd for fd in moodle_folders if mat in fd)
            # copy file
            folder = os.path.join(mpath, folder)
            folder_files = glob.glob(folder+"/*")
            for f2 in folder_files:
                os.remove(f2)
            copy_file(file, os.path.join(mpath, folder))
            n += 1
    print('%d files has been moved and copied to %s.' % (n, mpath))


def import_files(dpath, opath, suffix=None):
    if not os.path.exists(dpath):
        os.mkdir(dpath)
    folders = os.listdir(opath)
    n = 0
    for f in folders:
        afolder = os.path.join(opath, f)
        if os.path.isfile(afolder):
            continue

        files = os.listdir(afolder)
        if len(files) != 1:
            raise ValueError("Subfolder %s does not contain only one file, but %d files." % (f, len(files)))
        file = files[0]

        # rename it
        # use folder name
        _split = f.split('_')
        name = "_".join(_split[0].split(' '))
        # extract matricule
        mat = _split[2]
        name = name + "_%s_" % mat
        if suffix:
            name = name + suffix + ".pdf"
        else:
            name = name + file

        # copy file
        file = os.path.join(afolder, files[0])
        copy_file(file, os.path.join(dpath, name))
        n += 1
    print('%d files has been moved and copied to %s.' % (n, dpath))


def zipdirbatch(path, archive='moodle', batch=None):
    # ziph is zipfile handle
    i = 0
    j = 0
    narchive = archive
    ziph = zipfile.ZipFile(narchive+'.zip', 'w', zipfile.ZIP_DEFLATED)
    asize = 0  # archive size
    print("Compressing ", end="", flush=True)
    for root, dirs, files in os.walk(path):
        for file in files:
            i = i + 1
            pfile = os.path.join(root, file)
            ziph.write(pfile, os.path.relpath(pfile, path))  # add the file to the zip
            mbs = os.path.getsize(pfile) / MB  # file size in Mb
            asize += mbs
            if batch and asize >= batch:
                print('\nArchive %s.zip created.' % narchive)
                j = j + 1
                narchive = "%s%d" % (archive, j)
                ziph = zipfile.ZipFile(narchive + '.zip', 'w', zipfile.ZIP_DEFLATED)
                asize = 0
                print("Compressing ", end="", flush=True)
            else:
                print(".", end="" if i % 65 else "\nCompressing ", flush=True)
    print('\nArchive %s.zip created.' % narchive)
