# regex to find a matricule: 7 digits followed by not a number or the end of the line
re_mat = '[1-2]\\d{6}(?=(?:\\D|$))'

box = {
    "devoir": {
        'grade': (.1, .9, .6, .9),  # x1, x2, y1, y2 in % of the page width and height
        'trim': [(-1, 3)]  # i, n: n number of digits to remove at the end of the ith box
    },
    "exam": {
        'front': {
                      'id': (0.05, 0.84, 0.15, 0.3),
                      'grade': (0.86, 0.94, 0.22, 0.5)
                  },
        'matricule': (0.77, 0.95, 0.05, 0.105)
    }
}

