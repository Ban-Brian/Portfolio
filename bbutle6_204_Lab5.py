'''
-------------------------------------------------------------------------------
Name: Brian Butler 
Partner's Name: Yaw Ansong
Assignment: Lab 5
Due by end of your lab class
-------------------------------------------------------------------------------
Honor Code Statement: I received no assistance on this assignment that
violates the ethical guidelines set forth by the professor and class syllabus,
including any AI or code auto-complete tools.
-------------------------------------------------------------------------------
'''

#PART 1: Complete the Program
#NOTE: TEST YOUR FUNCTIONS!!!!!

#Task 1
def add_to_this(top):
    x = 1
    sum = 0
    while x <= top:
        sum = sum + x
        x += 1
    return sum

'''Identify: 
Control Variable: x
Initialization: x = 1
Modification: sum 
Describe the control:  The result that is returned from the function and when x is less than or equal to the top.
'''

#Task 2

def exp_by_hand(base, exp):
    next_exp = 1
    result = 1
    while next_exp <= exp:
        result = result * base
        next_exp += 1
    return result


'''Identify: 
Control Variable: next_exp
Initialization: next_exp = 1
Modification: next_exp += 1
Describe the control: As the next_exp is less than or equal to exp
'''


#PART 2: Write Your Own Function with Loops

#Task 3

def sum_divisors(n):
    total_sum = 0
    divisor = 1
    while divisor <= n:
        if n % divisor == 0:
            total_sum = total_sum + divisor
        divisor = divisor + 1
    return total_sum

'''Identify:
Control Variable: divisor
Initialization: divisor = 1
Modification: divisor + 1
Describe the control: if th divisor is less than or equal to n
'''
