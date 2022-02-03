import os
import argparse
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from process_copy import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Move the copy to moodle folders.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('path', type=str, help='path to the devoir folder')
    parser.add_argument('-i', '--import', default=False, action='store_true', dest='import_files',
                        help="Import files from moodle directories.")
    parser.add_argument('-e', '--export', default=False, action='store_true',
                        help="Export files from moodle directories.")
    parser.add_argument('-g', '--grade', type=str,
                        help="Read the grade on the first page of the pdf.\n"
                             "Need to define the box in config. Here the current configuration:\n"
                             "%s" % "\n".join(["  - \"%s\": %s" % (k, str(v)) for k, v in config.box.items()]))
    parser.add_argument('--grades', type=str,
                        help="Path to a csv file to add the grades. Needs columns \"Matricule\" and \"Note\".")
    parser.add_argument('-c', '--compare', default=False, action='store_true',
                        help="Compare grades found to the ones in the file provided in the member grades.")
    parser.add_argument("-b", "--batch", type=int,
                        help="Compress files by batches of the given size in Mb. Dafault: no batch.")
    parser.add_argument('-m', '--mpath', type=str, help='path to the moodle folders')
    parser.add_argument('-r', '--root', type=str, help='root path to add to all input paths')
    parser.add_argument("-s", "--suffix", type=str, help="Replace file name by this value when importing, "
                                                         "e.g. Devoir1_MTH1102_H22_Gr01.")
    parser.add_argument('-t', '--train', default=False, action='store_true', help='train the CNN on the MNIST dataset')
    args = parser.parse_args()

    if args.root:
        args.root = os.path.abspath(args.root)
        if not args.path.startswith('/'):
            args.path = os.path.join(args.root, args.path)
        if args.grades and not args.grades.startswith('/'):
            args.grades = os.path.join(args.root, args.grades)
        if args.mpath:
            if not args.mpath.startswith('/'):
                args.mpath = os.path.join(args.root, args.mpath)
        else:
            args.mpath = os.path.join(args.root, 'moodle')
    else:
        args.path = os.path.abspath(args.path)
        if args.grades:
            args.grades = os.path.abspath(args.grades)
        if args.mpath:
            args.mpath = os.path.abspath(args.mpath)
        else:
            args.mpath = os.path.join(args.path, 'moodle')

    if args.train:
        from process_copy.train import train
        train()

    if args.grade:
        from process_copy.grade import grade_all, compare_all, grade_all_exams
        if args.grade == 'devoir':
            if args.compare:
                compare_all(args.path, args.grades, config.box[args.grade])
            else:
                grade_all(args.path, args.grades, config.box[args.grade])
        elif args.grade == 'exam':
            grade_all_exams(args.path, args.grades, config.box[args.grade])
        else:
            raise ValueError("Grade configuration %s hasn't any action defined.")

    if args.import_files:
        from process_copy import mcc
        mcc.import_files(args.path, args.mpath, args.suffix)

    if args.export:
        from process_copy import mcc
        mcc.copy_files(args.path, args.mpath)
        mcc.zipdirbatch(args.mpath, archive=args.mpath, batch=args.batch)
