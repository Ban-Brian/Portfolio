'''
-------------------------------------------------------------------------------
Name: Brian Butler 
Partner's Name: Jonathan Regules
Assignment: Lab 6
Due by end of your lab class
-------------------------------------------------------------------------------
Honor Code Statement: I received no assistance on this assignment that
violates the ethical guidelines set forth by the professor and class syllabus,
including any AI or code auto-complete tools.
-------------------------------------------------------------------------------
'''
#PART 1: Complete the Program

#Task 1
def add_to_this(top):
    sum = 0
    for i in range (1, top + 1): #remove the placeholder blanks when you are ready to write your code
        sum = sum + i 
    return sum    


#Task 2
def exp_by_hand(base, exp):
    result = 1 #remove the placeholder blanks when you are ready to write your code
    for i in range(exp):
        result = result * base
    return result 


#PART 2: Write Your Own Function with for Loops

#Task 3

def add_every_third(l):
    total = 0
    for i in range(2, len(l), 3):
        num = l[i]
        if num > 0:
            total = total + num
    return total
        
        
    
