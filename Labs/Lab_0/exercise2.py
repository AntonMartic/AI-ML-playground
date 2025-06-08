
#can't use the ** operator, so creating the function:
def power(b, a):
    #handle power with 0 exponent
    if a == 0:
        return 1
    #handle power with 0 base:
    if b == 0 & a != 0:
        return 0
    #handle negative exponent
    if a < 0:
        result = 1
        for _ in range(-a): # loop number of times as |-a|
            result *= b # multiply result by b each time
        return 1 / result # b^-a = 1/b^a
    #handle positive base
    result = 1
    for _ in range(a): # loop number of times as |a|
        result *= b # multiply result by b each time
    return result

# Test function
assert power(1, 500) == 1
assert power(3, 4) == 81
assert power(0, 10) == 0
assert power(-2, 3) == -8
assert power(5, -2) == 0.04
assert power(0, 0) == 1

print(power(1,500))
print(power(3,4))
print(power(0,10))
print(power(-2,3))
print(power(5,-2))
print(power(0,0))