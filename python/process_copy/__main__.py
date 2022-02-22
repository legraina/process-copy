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
    parser.add_argument('path', type=str, help='path to the copies folder')
    parser.add_argument('-i', '--import', default=False, action='store_true', dest='import_files',
                        help="Import files from moodle directories.")
    parser.add_argument('-e', '--export', default=False, action='store_true',
                        help="Export files to moodle directories.")
    parser.add_argument('-f', '--find', type=str,
                        help="Find matricule in files according to the configuration provided. "
                             "Use moodle csv files to check existence of matricule if provided.\n"
                             "Need to define the boxes where to search for a matricule in config.py: "
                             "need a box for the front page and for a regular page. "
                             "Here the current configurations available:\n"
                             "%s" % "\n".join(["  - \"%s\": %s" % (k, str(v))
                                               for k, v in config.matricule_box.items()]))
    parser.add_argument('-g', '--grade', type=str,
                        help="Read the grade on the first page of the pdf according to the configuration provided.\n"
                             "Need to define the box in config. Here the current configurations available:\n"
                             "%s" % "\n".join(["  - \"%s\": %s" % (k, str(v)) for k, v in config.grade_box.items()]))
    parser.add_argument('--grades', type=str,
                        help="Path to a csv file to add the grades. Needs columns \"Matricule\" and \"Note\".")
    parser.add_argument('-c', '--compare', default=False, action='store_true',
                        help="Compare grades found to the ones in the file provided in the member grades.")
    parser.add_argument("-b", "--batch", type=int, default=500,
                        help="Compress files by batches of the given size in Mb. Default: 500 Mb.")
    parser.add_argument('-m', '--mpath', type=str, default='moodle', help='path to the moodle folders. Default: moodle')
    parser.add_argument('-r', '--root', type=str, help='root path to add to all input paths')
    parser.add_argument("-s", "--suffix", type=str,
                        help="Replace file name by this value when importing. "
                             "Default: {course}_{session}_{name} when any of them are defined.")
    parser.add_argument("-fp", "--frontpage", type=str,
                        help="Use the given latex file, fill it with the name and matricule, "
                             "then add it as a front page.")
    parser.add_argument("-na", "--name", type=str, help="Name of the devoir or exam.")
    parser.add_argument("-co", "--course", type=str, help="Name of the course.")
    parser.add_argument("-se", "--session", type=str, help="Name of the session.")

    parser.add_argument('-t', '--train', default=False, action='store_true', help='train the CNN on the MNIST dataset')
    args = parser.parse_args()

    if args.root:
        args.root = os.path.abspath(args.root)
        if not os.path.exists(args.root):
            raise ValueError('Root path %s does not exist.' % args.root)

    args.path = try_alternative_root(args.path, args.root, check=not args.import_files)
    args.mpath = try_alternative_root(args.mpath, args.root, check=args.import_files)
    args.frontpage = try_alternative_root(args.frontpage, args.root)
    args.grades = try_alternative_root(args.grades, args.root)

    l_input = ''
    suffix = ''
    if args.course:
        l_input += '\\renewcommand{\\cours}{%s}\n' % args.course
        suffix += '%s_' % args.course
    if args.session:
        l_input += '\\renewcommand{\\session}{%s}\n' % args.session
        suffix += '%s_' % args.session
    if args.name:
        l_input += '\\renewcommand{\\devoir}{%s}\n' % args.name
        suffix += args.name
    config.latex['input'] += l_input
    if args.suffix is None and suffix:
        args.suffix = suffix

    if args.train:
        from process_copy.train import train
        train()

    if args.find:
        from process_copy.grade import find_matricules
        find_matricules(args.path, config.matricule_box[args.find], args.grades)

    if args.grade:
        from process_copy.grade import grade_all, compare_all, grade_all_exams
        if args.grade == 'devoir':
            if args.compare:
                compare_all(args.path, args.grades, config.grade_box[args.grade])
            else:
                grade_all(args.path, args.grades, config.grade_box[args.grade])
        elif args.grade == 'exam':
            grade_all_exams(args.path, args.grades, config.grade_box[args.grade])
        else:
            raise ValueError("Grade configuration %s hasn't any action defined.")

    if args.import_files:
        from process_copy import mcc
        mcc.import_files(args.path, args.mpath, suffix=args.suffix, latex_front_page=args.frontpage)

    if args.export:
        from process_copy import mcc
        mcc.copy_files_for_moodle(args.path, args.mpath, args.grades)
        mcc.zipdirbatch(args.mpath, archive=args.mpath, batch=args.batch)
