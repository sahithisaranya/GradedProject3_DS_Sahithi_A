#Import required libraries and read dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Read the dataset
df = pd.read_csv('loan_approval_data.csv')
# Check the first few samples
print(df.head())

# Check the shape of the data
print(df.shape)

# Get the information about the data
print(df.info())
# Check for missing values
print(df.isnull().sum())

# Handle missing values
df = df.dropna()

# Drop redundant features if any
df = df.drop(['loan_id'], axis=1)
categorical_columns=df.select_dtypes(include=['object']).columns
print(categorical_columns)
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize loan status distribution with gender
sns.countplot(x='gender', hue='loan_status', data=df)
plt.show()

sns.countplot(x='married', hue='loan_status', data=df)
plt.show()

sns.countplot(x='education', hue='loan_status', data=df)
plt.show()

sns.countplot(x='self_employed', hue='loan_status', data=df)
plt.show()

sns.countplot(x='property_area', hue='loan_status', data=df)
plt.show()
# Encode categorical data
label_encoder = LabelEncoder()
categorical_features = ['gender', 'married', 'education', 'self_employed', 'property_area']
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

# Separate the target and independent features
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Build a classification model (example: Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model using pickle
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

import pymysql

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Step 8b: Register API
@app.route('/register', methods=['POST'])
def register():
    # Get username and password from request
    username = request.form['username']
    password = request.form['password']

    # Create a user object and store it in the database
    user = User(1, username, password)

    # Store the user details in the database using pymysql (you need to configure your MySQL connection)
    # ... code to store the user in the database ...

    return jsonify({'message': 'User registered successfully.'})

# Step 8c: Login API
@app.route('/login', methods=['POST'])
def login():
    # Get username and password from request
    username = request.form['username']
    password = request.form['password']

    # Check if the username and password are valid (you need to validate against the database)
    # ... code to validate the credentials ...

    return jsonify({'message': 'Login successful.'})

# Step 8d: Enter Details API
@app.route('/enter_details', methods=['GET'])
def enter_details():
    return render_template('predict.html')

# Step 8e: Predict API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user details from the form
    gender = request.form['gender']
    married = request.form['married']
    education = request.form['education']
    # ... get other details ...

    # Preprocess the user details
    user_details = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Education': [education]
        # ... add other details ...
    })

    # Encode the categorical features
    for feature in categorical_features:
        user_details[feature] = encoder.transform(user_details[feature])

    # Use the trained model to predict loan status
    prediction = model.predict(user_details)

    # Return the prediction results
    return render_template('prediction.html', prediction=prediction)

# Step 8f: Logout API
@app.route('/logout', methods=['POST'])
def logout():
    # Code to logout the user
    return jsonify({'message': 'Logout successful.'})

if __name__ == '__main__':
    app.run(debug=True)