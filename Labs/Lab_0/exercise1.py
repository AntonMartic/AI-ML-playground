import numpy as np

# 1. This is incorrect, lists are defined with brackets. Correct way:
list = [1,2]

# 2. This is correct.
tuple = (1,2)

# 3. This is incorrect, because pyhon interprets {} as a dictionary. Correct way:
empty_set = set()

# 4. This is incorrect and gives the result [1, 2, 3, 4]. Correct way:
l1 = [1, 2]
l2 = [3, 4]
plus = np.add(l1,l2)
#print(plus)

# 5. This is incorrect, because a string cannot concatenate "int". Correct way:
txt = "a" * 1000
#print(len(txt))

# 6. This is incorrect, because append does not work for multiple items. Correct way:
lst = [1, 2]
lst.extend([3, 4])
# print(lst)

# 7. This is correct.
list3x = [[]] * 3
# print(list3x)

# 8. This is incorrect, the string needs to be formated with f. Correct way:
X = 10
# print(f"x is equal to {X}")

# 9. This is correct
t9 = "ABCD"[::-1]
# print(t9)

# 10. This is incorrect and gives the result BCDE. Correct way:
t10 = "ABCDEF"[1:]
# print(t10)

# 11. This is incorrect and the result is false. Correct way:
t11 = "12345678"[1::2]
#print(t11)

# 12. This is incorrect and the result is false because python does not count duplicates in sets. Correct way:
t12 = len([1, 1, 2])
# print(t12)

# 13. This is correct.
t13 = dict()
# print(t13)

# 14. This is incorrect, the string needs to be formated with f. Correct way:
# print(f"a\nb")

# 15. This is incorrect, can for example mix and match with ''. Correct way:
#print('a"b')

# 16. This is incorrect. Correct way:
#print("\\\\\\\\")

# 17. This is correct.
#print("a\bc")

# 18. This is incorrect, range needs to go to 6. Correct way:
#for i in range(2, 6):
#    print(i)

# 19. This is correct, scipy is built on top of NumPy
#import scipy

# 20. This is correct.
#import numpy as np

# 21. This is incorrect, second import overwrites the first. Correct way:
#import math
#import numpy as np

# 22. This is incorrect, bad practice that brings all of numpy’s functions,
# classes, and constants into the global namespace.
# Can lead to conflicts, especially if other libraries in the code use the same names. Correct way:
# import numpy as np

# 23. This is correct.
# #import numpy as np

# 24. This is incorrect, because (pi) is explicitly renaimed to the alias (PI). Correct way:
#from numpy import pi, pi as PI

# 25. This is incorrect, nan is not equal to anything, including itself, can not satisfy the statement

# 26. This is incorrect, it is equal to 0.0. np.nan can not be equal to anything

# 27. This is correct.
#print(-1 * np.inf)

# 28. This is incorrect. Correct way:
#print(2 / 3)

# 29. This is incorrect, and the size is 6. Correct way:
arr = np.zeros((1, 2))
#print(arr.size == 2)

# 30. This is incorrect, int used and not float. Correct way:
#print(0.0 / 0.0)

# 40. This is incorrect, it renders the plot. Correct way:
#plt.savefig()

# 41. This is correct.
#print(type(2**3))

# 42. This is correct.
#print(np.zeros((3, 2)))

# 43. This is incorrect, gives a size of 30. Correct way:
#arr = np.zeros((3, 2, 5))
#print(arr.shape)

# 44. This is incorrect, it should be ndim. Correct way:
arr = np.zeros((3, 2))
arr_reshaped = arr.reshape((6,))
#print(arr_reshaped.ndim == 1)

# 45. This is incorrect, classes are also objects
class Book:
    pass
#print(type(Book))

