'''
Name: Brian Butler 
Assignment: PA2 – Functions and Branching
Due Date: 9/29/2025
Academic Integrity Statement: I received no assistance on this assignment that
violates the ethical guidelines set forth by the professor, university,
or the class syllabus, including AI or other prohibited sources.
'''

# I kept on getting errors with it saying I was using an illegle thing in my code but I was unsure at to what was causing the issue
# and when I deleted my comments on the progress it worked, I'm unsure hwo the auto grader works when grading that.
# I have no idea what my error is or why I am getting it.


def discountEligibility(age, isMember):  # I just set the parameters to fit the ranges of what was givent to me.
    if age <= 12:
        return "Child Discount"
    elif age > 65:
        return "Senior Discount"
    elif isMember:
        return "Membership Discount" # I use the if/elif/else statements to provide the ranges and use it from there.
    else:
        return "No Discount"
def assignAccess (isStaff, hasID, emergency, department):
    if isStaff and hasID:
        return "Full Access"
    elif isStaff:
        return "Temporary Access"
    elif emergency or department == "ER":  # This has a similar idea and reminds me a bit of discrete math.
        return "Emergency Access"
    elif hasID:
        return "Visitor Access"
    else:
        return "No Access"
def prioritizeTreatment(level, age, contagious, pregnant):
    if level>= 8 and contagious:
        return "Critical Priority"
    elif (age >= 65 and level >= 6) or pregnant: # This statement took me a while but im finding the rnage beteen 65 and 6.
        return "High Priority"
    elif level >= 4 or (age < 3 and contagious):
        return "Medium Priority"
    else:
        return "Low Priority" # I hand wrote out all the different ways it could go to see it easier 

def defineBillCategory(age, insurance, income, disability, veteranStatus):
    if insurance:
        if income < 30000: # I start off with definfing the people who would be eligible to even meet the standards for insurance the most.
            if disability or veteranStatus:
                return "Minimal Co-Pay"
            else:
                return "Low Co-Pay"
        elif 30000 <= income < 70000: # Then I move up to the people that would qualify for less assistance with their exceptions 
            if age > 65:
                return "Reduced Standard Co-Pay"
            else:
                return "Standard Co-Pay"
        else: 
            return "High Co-Pay"
    else: 
        if age < 18:
            return "Government Covered"
        elif disability or veteranStatus: 
            return "Government Assistance Program"  # These groups would be the exceptions that wouldn't fall into the traditional categories. 
        elif income < 20000:
            return "Subsidized Payment Plan"
        else:
            return "Full Payment Required"
def calculateBill (days, roomType, services):    
    total_bill = 0.0 # I intialize where it would start off to as the day 0 = 0

    if roomType == 'general':
        total_bill += days * 200
    elif roomType == 'semi-private':
        total_bill += days * 400
    elif roomType == 'private':
        total_bill += days * 700 # I segment off the price of each room and assign then each to different rates.

    if services == 'lab':
        total_bill += 300
    elif services == 'surgery':
        total_bill += 1000 # I define the services next that are able to be tacked onto the stay prices.

    if days > 10:
        total_bill *= 0.8 # This a discount of if you stay over 10 days and shold be the last thing needed.

    return total_bill
