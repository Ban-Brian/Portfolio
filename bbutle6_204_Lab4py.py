'''
-------------------------------------------------------------------------------
Name: Brian Butler
Partner's Name: Jonathan Regules
Assignment: Lab 4
Due by end of your lab class
-------------------------------------------------------------------------------
Honor Code Statement: I received no assistance on this assignment that
violates the ethical guidelines set forth by the professor and class syllabus,
including any AI or code auto-complete tools.
-------------------------------------------------------------------------------
'''

# Part 1 -- Understanding Branching 

def branches_function():
    x = 1
    result = False
    if (x >= 0):
        result = True
    return result

'''
Run the function above. What does it always return? Why?

Answer: The original function always returns False. This is because the variable x is set to 1. The condition checks if x is both
less than zero and not equal to zero. Since 1 is not less than 0, the condition evaluates to False,
the code inside the if statement is skipped, 
and the function returns the initial value of 'result', which is Falese.

Change the function so that it will always return True. Explain what you changed
and why it will now always return True.

Answer: We changed the if statement to be greater than or equal to 0 in order
for the value of 1 to always be evaluted as greater than 0 staiisfying the if
statement. 

'''

# Part 2 -- Writing Code with Branching

def categorize(price, units_sold):
    if price < 10 and units_sold < 10:
        return "Low-priced"
    elif price >= 10 and price <= 20 and units_sold < 20:
        return "Mid-priced"
    elif price > 20 and units_sold < 30:
        return "High-priced"
    elif price > 20 and units_sold >= 30:
        return "Premium"
    else:
        return "Other"


    




