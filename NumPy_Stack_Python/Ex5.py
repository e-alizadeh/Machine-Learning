import numpy as np


def is_symmetric_numpyFunc(matrix):
    if (matrix.transpose() == matrix).all():
        return True
    else:
        return False
 
def is_symmetric_noNumPy(matrix):
    # Assumption is that the matrix is of type 'list'
    dim = matrix.shape[0]  
    for i in range(dim):
        for j in range(dim):
            if i==j:
                continue
            if matrix[i,j] != matrix[j,i]:
                tempFLAG = True
                break
        if tempFLAG:
            return True 
            break





A = np.array([ [1,2,3], [2,4,5],[3,5,6] ])
B = np.array([ [1,2,3], [3,4,5],[3,5,6] ])

# Using Numpy transpose() function
matrix = A;
(matrix.transpose() == matrix).all() # Test whether all array elements along a given axis evaluate to True
# OR 
np.allclose(matrix,matrix.T) == 1 # allclose check whether the entities are equal to each other within in small tolerance

########################################################################	
# https://terribleatmaths.wordpress.com/2013/09/20/python-check-if-symmetric/
# A list is symmetric if the first row is the same as the first column,
# the second row is the same as the second column and so on. Write a
# procedure, symmetric, which takes a list as input, and returns the
# boolean True if the list is symmetric and False if it is not.
def symmetric(x):
    # First check that no lists within
    # x contain more values than the
    # total number of lists in x
    length = len(x)
    for value in range(0,length):
        if len(x) != len(x[value]):
            return False
    # Sets a counter to zero and
    # creates two empty lists, which
    # will be compared with one another.
    checkcolumn = 0
    list1 = []
    list2 = []
    # Whilst the counter remains lower
    # than the length of x, add the values
    # in the nth list to list1 and add
    # the values in the nth column to list2.
    # If the two lists are identical, the loop
    # continues. If the loop completes, then
    # x is symmetric.
    while checkcolumn < len(x):
        list1+=x[checkcolumn]
        for row in x:
            list2.append(row[checkcolumn])
        if list1 != list2:
           return False
        else:
            checkcolumn+=1
    return True