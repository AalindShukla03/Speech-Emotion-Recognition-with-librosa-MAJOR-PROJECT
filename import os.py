import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Change the current working directory
new_directory = r'D:\MAJOR PROJECT\major project complete'
os.chdir(new_directory)

# Print the updated current working directory
print("Updated Working Directory:", os.getcwd())
