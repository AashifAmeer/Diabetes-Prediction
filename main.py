import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Splitting features and labels
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Making predictions on new data
new_data = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [130],
    'BloodPressure': [70],
    'SkinThickness': [35],
    'Insulin': [180],
    'BMI': [25],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [30]
})

new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)

if new_prediction[0] == 1:
    print("The person is predicted to have diabetes.")
else:
    print("The person is predicted not to have diabetes.")
