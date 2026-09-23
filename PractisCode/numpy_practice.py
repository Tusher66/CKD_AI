import numpy as np

ages = np.array([16,43,86,12,34,49])

print("Average age", ages.mean())
print("Max Age",ages.max())
print("Min Age",ages.min())

# pataints = np.array(
#     [12,45],
#     [3,76],
#     [32,54]
# )

# print("Patation",pataints)

# print("Patien shape",pataints.shape)

patients = np.array([
    [25,80],
    [40,95],
    [65,100]
])

print(patients)

print("Shape:", patients.shape)

matrix = patients.reshape(2,3)
print(matrix)

a = np.array([12,45,89])
b = np.array([43,76,45])

print(a+b)
print(a-b)
print(a*b)
print(a/b)