# Greet User Function

def greet_user(name):
    print(f"Hello, {name}! Welcome to Python learning.")

# Example usage
greet_user("Zeba")

#  Print Even Numbers from List

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Even numbers in the list:")
for num in numbers:
    if num % 2 == 0:
        print(num)

# Student Data Dictionary

student = {
    "name": "Zeba",
    "age": 20,
    "course": "Computer Science"
}

print("Student Information:")
for key, value in student.items():
    print(f"{key.capitalize()}: {value}")

# Calculator with Functions

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b != 0:
        return a / b
    else:
        return "Cannot divide by zero!"

# Example usage
x = 10
y = 5
print("Add:", add(x, y))
print("Subtract:", subtract(x, y))
print("Multiply:", multiply(x, y))
print("Divide:", divide(x, y))
    