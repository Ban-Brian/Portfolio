def add_every_third(l):
    total = 0
    for i in range(2, len(l), 3):
        num = l[i]
        if num > 0:
            total = total + num
    return total
