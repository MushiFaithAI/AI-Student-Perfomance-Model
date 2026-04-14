import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Features (X) and target (y)
X = data[["hours_studied"]]
y = data["exam_score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("=== MODEL EVALUATION ===")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict example
example = [[6]]
prediction = model.predict(example)

print("\n=== PREDICTION ===")
print("If a student studies 6 hours, predicted score is:", prediction[0])