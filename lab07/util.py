import constants as c

def add_if_distinct(roots_list, candidate):
    for r in roots_list:
        if abs(r - candidate) <= c.EPSILON:
            return
    roots_list.append(candidate)
