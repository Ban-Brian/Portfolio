'''
-------------------------------------------------------------------------------
Name: Brian Butler 
Partner's Name: Yaw Ansong
Assignment: Lab 3
Due by end of your lab class
-------------------------------------------------------------------------------
Honor Code Statement: I received no assistance on this assignment that
violates the ethical guidelines set forth by the professor and class syllabus,
including any AI or code auto-complete tools.
-------------------------------------------------------------------------------
'''

#Partner A -- write area_of_base function here

def area_of_base(length):
    square_area = length * length
    return round(square_area, 4)

#Partner B -- write area_of_side function here

def area_of_side(base, height):
    side_area =(1/2)*base*height
    return round(side_area, 4)

#Write pyramid_surface_area function to calculate surface area of pyramid here

def pyramid_surface_area(base_side, slant_height):
    surface_area = base_side**2 + (2 * base_side * slant_height)
    return round(surface_area, 4)

#Write pyramid_volume function to calculate the volume of the pyramid (and any other functions you make) here

def pyramid_volume(base_side, height):
    base_area = base_side**2
    volume_pyramid = (1/3)*base_area*height
    return round(volume_pyramid, 4)
    
