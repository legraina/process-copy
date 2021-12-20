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
    parser.add_argument('-i', '--import', default=False, action='store_true',
                        help="Import files from moodle directories.")
    parser.add_argument('-e', '--export', default=False, action='store_true',
                        help="Export files from moodle directories.")
    parser.add_argument('-g', '--grade', type=str,
                        help="Read the grade on the first page of the pdf.\n"
                             "Need to define a box or use a predetermined one among:\n"
                             "%s" % "\n".join(["  - \"%s\": %s" % (k, str(v)) for k, v in config.box.items()]))
    parser.add_argument('--grades', type=str,
                        help="Path to a csv file to add the grades. Needs columns \"Matricule\" and \"Note\".")
    parser.add_argument('-c', '--compare', default=False, action='store_true',
                        help="Compare grades found to the ones in the file.")
    parser.add_argument("-b", "--batch", type=int, help="Compress files by batches.")
    parser.add_argument('-m', '--mpath', type=str, help='path to the moodle folders')
    parser.add_argument('-t', '--train', default=False, action='store_true', help='train the CNN on the MNIST dataset')
    args = parser.parse_args()

    args.path = os.path.abspath(args.path)
    if args.mpath:
        args.mpath = os.path.abspath(args.mpath)
    else:
        args.mpath = os.path.join(args.path, 'moodle')

    if args.train:
        from process_copy import grade
        grade.train()

    if args.grade:
        if args.grade.startswith('('):
            box = [float(s) for s in args.grade[1:-1].split(',')]
        else:
            box = config.box[args.grade]
        from process_copy import grade
        grade.grade_all(args.path, args.grades, box, args.compare)

    if args.export:
        from process_copy import mcc
        mcc.copy_files(args.path, args.mpath)
        if args.grades:
            mcc.copy_file(args.grades, args.mpath)
        mcc.zipdirbatch(args.mpath, batch=args.batch)
