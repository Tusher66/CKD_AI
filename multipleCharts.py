import matplotlib.pyplot as plt
import pandas as pd

# x = [1,2,3,4,5,6]
# y = [10,15,30,20,50,60]

# plt.plot(x,y)

# plt.show()

# days = [1,2,3,4,5]

# blood_pressure  = [120,150,180.130,145]

# plt.plot(days,blood_pressure)
# plt.title("Blood pressure Data")
# plt.xlabel("Blood pressure")
# plt.ylabel("Days")
# plt.show()

# days = [1,2,3,4,5]

# blood_pressure = [120,122,119,124,121]

# plt.plot(days,blood_pressure,
#          linestyle="--",
#     marker="^")

# plt.title("Blood Pressure Over Time")

# plt.xlabel("Day")

# plt.ylabel("Blood Pressure")

# plt.show()

# patients = [
#     "Ali",
#     "Hasan",
#     "Karim"
# ]

# creatinine = [
#     1.2,
#     2.8,
#     0.9
# ]

# plt.bar(patients,creatinine)

# plt.title("Creatinine Levels")

# plt.show()

# labels = [
# "CKD",
# "Healthy"
# ]

# values = [
# 180,
# 220
# ]

# plt.pie(
#     values,
#     labels=labels,
#     autopct="%1.1f%%"
# )

# plt.title("Patient Distribution")

# plt.show()

# plt.figure(figsize=(8,5))

# plt.plot(
#     [1,2,3],
#     [4,5,6]
# )

# plt.grid(True)

# plt.show()

# df = pd.read_csv("dataset/ckd.csv")

# plt.hist(
#     df["Age"],
#     bins=10
# )

# plt.title("Age Distribution")

# plt.xlabel("Age")

# plt.ylabel("Patients")

# plt.show()


df = pd.read_csv("dataset/ckd.csv")

# Histogram
plt.hist(df["Age"])

plt.title("Age Distribution")

plt.show()

# Scatter
plt.scatter(
    df["Age"],
    df["SerumCreatinine"]
)

plt.xlabel("Age")

plt.ylabel("Creatinine")

plt.show()

# Bar
plt.bar(
    df["PatientID"],
    df["BMI"]
)

plt.title("Blood Pressure")

plt.show()

plt.savefig("filename.png")