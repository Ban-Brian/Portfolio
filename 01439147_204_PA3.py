# Name: Brian Butler
# Assignment 3
# Due Date: 10/13
# Honor Code Statement: I received no assistance on this assignment that
# violates the ethical guidelines set forth by the professor, university,
# or the class syllabus, including AI or other prohibited sources.

##########    Task 1     ####################

def add_em_up(b, t):
    total = b       # start total with bottom number
    count = 0       # count how many numbers we add
    add = 1         # first number to add
    # keep adding while total + next number <= top
    while total + add <= t:
        total = total + add   # add next number to total
        count = count + 1     # increase count
        add = add + 1         # move to next number

    return count   # return how many numbers were added

##########    Task 2     ####################

def add_em_up2(n1, n2):
    # call add_em_up with smaller number first
    if n1 < n2:
        return add_em_up(n1, n2)
    else:
        return add_em_up(n2, n1)

##########    Task 3     ####################

def zoom_boom(nums, pin):
    out = []   # list to store results

    for n in nums:
        if n % pin == 0 and n % 10 == pin:
            out.append("Zoom")    # divisible by pin and ends with pin
        elif n % 10 == pin:
            out.append("Boom")    # ends with pin only
        elif n % pin == 0:
            out.append("Zoom")    # divisible by pin only
        else:
            out.append(n)         # leave number as is

    return out   # return the final list

##########    Task 4     ####################

def elem_loc(lst, target):
    for i in range(len(lst)):
        if lst[i] == target:
            return i   # return index if target found
    return None       # return None if target not found

##########    Task 5     ####################

def update_credits(stu_data, new_data):
    # go through new_data two items at a time (ID, credits)
    for i in range(0, len(new_data), 2):
        sid = new_data[i]       # student ID
        creds = new_data[i+1]   # number of credits
        pos = elem_loc(stu_data, sid)  # check if student already exists

        if pos is not None:
            stu_data[pos+1] += creds  # add credits if student exists
        else:
            stu_data.append(sid)      # add new student ID
            stu_data.append(creds)    # add new credits

    return stu_data   # return updated student list

##########    Task 6     ####################

def clean_credits_list(messy_list):
    res = []   # list to store clean data

    for pair in messy_list:
        sid = pair[0]   # student ID
        cr = pair[1]    # credits
        idx = None      # will hold index if student exists
        j = 0
        # check if student ID is already in res
        while j < len(res):
            if res[j] == sid:
                idx = j   # save index of existing student
            j += 2       # move to next student ID

        if idx is not None:
            res[idx + 1] += cr  # add credits if student already exists
        else:
            res.append(sid)     # add new student ID
            res.append(cr)      # add their credits

    return res   # return the cleaned list
