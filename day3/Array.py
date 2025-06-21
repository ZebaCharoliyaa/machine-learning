#  Creating and Manipulating Arrays

import numpy as np

# Create 1D array with elements from 10 to 50
arr = np.arange(10, 51)
print("Original array:", arr)

# Extract elements from index 5 to 15
print("Elements from index 5 to 15:", arr[5:16])

# Reverse the array
print("Reversed array:", arr[::-1])

# Print every 3rd element
print("Every 3rd element:", arr[::3])



# 2D Array Indexing

# Create a 3×3 matrix with values 1 to 9
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("3x3 Matrix:\n", matrix)

# Extract the second row
print("Second row:", matrix[1])

# Extract the first column
print("First column:", matrix[:, 0])

# Extract a 2×2 submatrix from the top-left corner
print("2x2 submatrix from top-left corner:\n", matrix[:2, :2])



#  Arithmetic Operations

# Create two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition
print("Addition:", a + b)

# Subtraction
print("Subtraction:", a - b)

# Multiplication
print("Multiplication:", a * b)

# Division
print("Division:", a / b)