import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('bank_customer_churn.csv')

# Define features (X) and target variable (y)
X = data[['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']]
y = data['churn']

# Define preprocessing transformers for numerical and categorical features
numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
categorical_features = ['country', 'gender', 'credit_card', 'active_member']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

# Create a preprocessor that applies transformers to the respective feature groups
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that preprocesses data and trains a classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

X_preprocessed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Create and train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Example prediction for a new customer
new_customer_data = pd.DataFrame({
    'credit_score': [750],
    'country': ['France'],
    'gender': ['Male'],
    'age': [30],
    'tenure': [5],
    'balance': [1500],
    'products_number': [2],
    'credit_card': [1],
    'active_member': [1],
    'estimated_salary': [60000]
})

# Preprocess the new data
new_customer_preprocessed = pipeline.transform(new_customer_data)

# Make a churn prediction for the new customer
churn_prediction = model.predict(new_customer_preprocessed)

if churn_prediction[0] == 1:
    print('The new customer is predicted to churn.')
else:
    print('The new customer is predicted to stay.')
