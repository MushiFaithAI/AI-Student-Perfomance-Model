import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Show data
print("DATA PREVIEW")
print(data.head())

# Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Statistics
print("\nSTATISTICS")
print(data.describe())

# NumPy analysis
print("\nAVERAGE EXAM SCORE")
print(np.mean(data["exam_score"]))

# Visualization
plt.scatter(data["hours_studied"], data["exam_score"])
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()