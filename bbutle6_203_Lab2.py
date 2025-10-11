 '''
-------------------------------------------------------------------------------
Name: Brian Butler 
 Partner's Name: Yaw Ansong
Assignment: Lab 2
 Date: 9/4/25
 Due by end of your lab class
 -------------------------------------------------------------------------------
 Honor Code Statement: I received no assistance on this assignment that
violates the ethical guidelines set forth by the professor and class syllabus,
 including any AI or code auto-complete tools.
 -------------------------------------------------------------------------------
 '''
 
 '''
 ###################################################
 Your OWN test cases:
 1) Barbie: 1 Ken: 2 Hours: 1/6
   Results: 30
 
2) Barbie:10 Ken: 5 Hours: 10/6
    Results: 1500

 3) Barbie: 3  Ken: 7 Hours: 1
   Results: 600

4) Barbie: 18 Ken: 16 Hours: 2
   Results: 4080
5) Barbie: 10000 Ken: 100000  Hours: 1
    Results: 6600000
 '''


 '''
 ##############################
 Write out the algorithm (your plan) for the task:

First we need to assign what our input is to a variable. Then we add the two separate variables together. After that we multiply to get the number of lights the put up in a set period of time.
'''


#Type your code below 


Barbie = int(input("Enter hanging speed of Barbie (lights per minute): "))
Ken = int(input("Enter hanging speed of Ken (lights per minute): "))
total_time = int(input("Enter total time in minutes: "))

combined_speed = Barbie + Ken

total_lights = combined_speed * total_time

print("The total number of lights hung are:", total_lights)



 







 

