print("Hello Tusher CKD AI")

name = "Tusher"
age = 23
weight = 72
height = 1.6764   # Height in meters (5'6")
is_student = True

print(name)
print(age)
print(height)
print(is_student)

numbers = [12, 34, 534, 54, 34]

for item in numbers:
    print(item)

if age < 18:
    print(age)
else:
    print("Age is high")


def bmi(h, w):
    return w / (h ** 2)


result = bmi(height, weight)
print("BMI:", round(result, 2))