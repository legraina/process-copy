# regex to find a matricule: 7 digits followed by not a number or the end of the line
re_mat = '[1-2]\\d{6}(?=(?:\\D|$))'

grade_box = {
    "devoir": {
        'grade': (.1, .9, .6, .9),  # x1, x2, y1, y2 in % of the page width and height
        # 'trim': [(-1, 3)]  # i, n: n number of digits to remove at the end of the ith box
    },
    "exam": {
        'grade': (0.86, 0.94, 0.22, 0.5)
    }
}

matricule_box = {
    "intra": {
        'front': (0.05, 0.85, 0.2, 0.35),
        'regular': (0.55, 0.95, 0.05, 0.13),
        'separate_box': True  # True if there are some boxes for each digit of the matricule
    }
}

latex = {
    'cmd': "pdflatex",
    'input-file': 'data.tex',
    'input': "\\renewcommand{\\nom}{%s}\n\\renewcommand{\\matricule}{%s}\n"
}
