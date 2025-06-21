#  Create a list of all vowels from a string using list comprehension

input_str = "Python Programming is fun!"
vowels = [char for char in input_str if char.lower() in 'aeiou']
print("Vowels in the string:", vowels)

# Generate a list of numbers from 1 to 50 divisible by 3 and 5

divisible_by_3_and_5 = [num for num in range(1, 51) if num % 3 == 0 and num % 5 == 0]
print("Numbers divisible by 3 and 5:", divisible_by_3_and_5)

#  Create a list of tuples containing number and its square for numbers from 1 to 10

squares = [(num, num**2) for num in range(1, 11)]
print("Number and its square:", squares)