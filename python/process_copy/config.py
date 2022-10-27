# regex to find a matricule: 7 digits followed by not a number or the end of the line
re_mat = '[1-2]\\d{6}(?=(?:\\D|$))'

grade_box = {
    "devoir": {
        'grade': (.1, .9, .6, .9),  # x1, x2, y1, y2 in % of the page width and height
        # 'trim': [(-1, 3)]  # i, n: n number of digits to remove at the end of the ith box
    },
    "exam": {
        'grade': (0.8, .95, 0.2, 0.55),
        # i, n: n number of digits to remove at the end of the ith box. -1 means to trim everything
        # 'trim': [(0, -1), (1, 2), (2, 2), (3, 3), (4, 2), (5, 3)]
    }
}

matricule_box = {
    "exam": {
        'front': (0.05, 0.85, 0.15, 0.35),
        'regular': (0.55, 0.95, 0.05, 0.13),
        'separate_box': True  # True if there are some boxes for each digit of the matricule
    }
}

known_mistmatch = {7: 1}


class Latex:
    cmd = "pdflatex"
    input_file = 'data.tex'
    input_content = "\\renewcommand{\\nom}{%s}\n\\renewcommand{\\matricule}{%s}\n"


class MoodleFields:
    mat = 'Matricule'
    name = 'Nom complet'
    id = 'Identifiant'
    grade = 'Note'
    max = 'Note maximale'
    mdate = 'Dernière modification (note)'
    status = 'Statut'
    status_start_filter = 'Remis'
    group = 'Groupe'
