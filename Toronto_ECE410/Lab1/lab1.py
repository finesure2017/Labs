"""
This file is the pre-requisite to the course ECE410: Control Systems at University of Toronto
It is the basics to working with Linear Algebra and Ordinary Differential Equations
"""
import numpy as np

""" Create the matrix
  1 2 3 
[ 4 5 6 ]
  7 8 9
"""

def createMatrix(row, col):
    result = np.empty((row, col))
    count = 1
    for rowIndex in range(row):
        for colIndex in range(col):
            result[rowIndex, colIndex] = count
            count += 1
    return result 

def createMatrixWithValue(row, col, value):
    return value * np.ones((row, col))

def KthRow(matrix, k):
    return matrix[k][:]

def KthCol(matrix, k):
    return matrix[:,k]

if __name__ == "__main__":
    A = createMatrixWithValue(3, 3, 2)
    print A
    A = createMatrix(3, 3)
    print A
    print KthRow(A, 0)
    print KthCol(A, 0)
    # Make Reduced Row Echelon Form Manually
    A[2][:] -= 7 * A[0][:]
    A[1][:] -= 4 * A[0][:]
    A[2][:] -= 2*A[1][:]
    A[1][:] /= -3
    A[0][:] -= 2 * A[1][:]
    print A







