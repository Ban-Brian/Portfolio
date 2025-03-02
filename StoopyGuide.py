import math
# Question 1

# Global Variables: Variables defined outside any function but are available throughout the entire program.

# Former Parameter Variables: Variables defined within a function's definition, which are used to
# accept input values when the function is called.

#  Local Variables: Variables declared within a function, only accessible within that function.

# Question 2

def hypot(a, b):
    return math.sqrt(a**2 + b**2)

print(hypot(3, 4))

# The (def hypot(a, b):) is taking two arguments – a and b –  and calculating the hypotenuse. YOU NEED TO IMPORT MATH,
# without it the function, math.sqrt,  doesn't work.
#  The –print(hypot(3, 4))– is calling the function to do the √3^2 + 4^2 problem. then printing result.

# Question 3
def sumFifths(n):
    total = 0
    for i in range(0,n,5):
        total = total + 1
    return total
print(sumFifths(20))

# Think of the accumulator as the score-keeper in a game.
# In this function the accumulator variable would be (total) because an accumulator variable
# is used to store and update a running total or cumulative value during a loop.

# Question 4
def qNotU(s):
    s = s.lower()
    if 'q' in s and 'u' not in s:
        return True
    return False

print(qNotU("Que"))
print(qNotU("unique"))
print(qNotU("queue"))
print(qNotU("Qatar"))

# the function – def qNotU(s): – is defined with one parameter being "s", for string.
# to convert it to lower case you need to use s = s.lower() which ensures the the check for q and u is case-insensitive.
# For this function:
# –  if 'q' in s and 'u' not in s:
#         return True
#     return False –
# You need to use an "if" statement to check for two conditions being:
# 'q' in s : this checks if q is present in string
# 'u' in s : this checks if  u  is not in the string
# If both conditions are met then the statement would return as "True"
# If either or both conditions aren't then it would return as "False"

# Question 5
 # “lo” in “hello” and “bye” in “goodbye” or not 3+2 > 7, should look like:
print(("lo" in "hello" or "bye" in "goodbye") or (3 + 2 > 7))

# It should print -   Answer:   `True`
#  `"lo" in "hello"` evaluates to `True`
# `"bye" in "goodbye"` evaluates to `True`
# `3 + 2 > 7` evaluates to `False`, so `not 3 + 2 > 7` evaluates to `True`
# The expression simplifies to `True and True or True` which evaluates to `True`
