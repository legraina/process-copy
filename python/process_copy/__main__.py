import os
import argparse
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from process_copy import config


def try_alternative_root(path, root=None, check=True):
    if not path:
        return None

    if path.startswith('/'):
        return path

    if root:
        npath = os.path.join(root, path)
        # if file exist, return new path
        if not check or os.path.exists(npath):
            return npath

    # try path as a relative path
    npath = os.path.abspath(path)
    # if file doesn't exist throw an error
    if check and not os.path.exists(npath):
        raise ValueError('Path %s does not exist.' % path)

    return npath


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
    parser.add_argument("-b", "--batch", type=int, default=500,
                        help="Compress files by batches of the given size in Mb. Default: 500 Mb.")
    parser.add_argument('-m', '--mpath', type=str, default='moodle', help='path to the moodle folders. Default: moodle')
    parser.add_argument('-r', '--root', type=str, help='root path to add to all input paths')
    parser.add_argument("-s", "--suffix", type=str, help="Replace file name by this value when importing, "
                                                         "e.g. Devoir1_MTH1102_H22_Gr01.")
    parser.add_argument("-f", "--frontpage", type=str,
                        help="Use the given latex file, fill it with the name and matricule, "
                             "then add it as a front page.")
    parser.add_argument('-t', '--train', default=False, action='store_true', help='train the CNN on the MNIST dataset')
    args = parser.parse_args()

    if args.root:
        args.root = os.path.abspath(args.root)

    args.path = try_alternative_root(args.path, args.root)
    args.mpath = try_alternative_root(args.mpath, args.root, check=False)
    args.frontpage = try_alternative_root(args.frontpage, args.root)
    args.grades = try_alternative_root(args.grades, args.root)

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
        mcc.import_files(args.path, args.mpath, suffix=args.suffix, latex_front_page=args.frontpage)

    if args.export:
        from process_copy import mcc
        mcc.copy_files_for_moodle(args.path, args.mpath, args.grades)
        mcc.zipdirbatch(args.mpath, archive=args.mpath, batch=args.batch)
