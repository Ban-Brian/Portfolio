#Problem 1
def find_octahedron_mensuration(side):
    side_length = int(side)
    
    sqrt_3 = pow(3, 0.5)
    sqrt_2 = pow(2, 0.5)
    
    a_raw = 2 * sqrt_3 * pow(side_length, 2)
    v_raw = (sqrt_2 / 3) * pow(side_length, 3)
    
    a_rounded = round(a_raw, 2)
    v_rounded = round(v_raw, 2)
    
    result = f"area:{a_rounded}, volume:{v_rounded}"
    
    return result

#Problem 2
def is_even(dec_num):
    num = int(dec_num)
    
    b_str = bin(num)
    
    lsb = b_str[-1]
    
    if lsb == '0':
        parity = "even"
    else:
        parity = "odd"
    output = f"{b_str} is {parity}."

    return output

#Problem 3

def encrypt_word(plain_word, key):
    res = []
    
    for c in plain_word:
        v = ord(c)
        v += key
        new_c = chr(v)
        res.append(new_c)
      
    cipher = "".join(res)
    
    return cipher
