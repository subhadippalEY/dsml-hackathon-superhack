import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Custom transformer to replace inf/-inf with NaN
class ReplaceInf(BaseEstimator, TransformerMixin):
    def __init__(self, value=np.nan):
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], self.value)
        return X
    
# Load your dataset
# Assuming the dataset is in CSV format and stored at 'software_quality_data.csv'
df = pd.read_csv('C:\\Users\XZ957MG\\source\\repos\\IIIMRoorkie-Assignment\\IIIMRoorkie-Assignment\\Hackathon\\train_data1.csv')

# Splitting the dataset into features and target variable
X = df.drop(['id', 'defects'], axis=1)
y = df['defects']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size = 0.2, random_state=42) #test_size=32056, train_size=74794, random_state=42)

# Identifying numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

# Creating transformers for numerical and categorical data
#numerical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='mean')),
#    ('scaler', StandardScaler())
#])

# Update the numerical_transformer to include ReplaceInf
numerical_transformer = Pipeline(steps=[
    ('replace_inf', ReplaceInf()),  # Add this line to replace inf/-inf with NaN
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_string', FunctionTransformer(lambda x: x.astype(str), validate=False)),  # Convert to string
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print(X_train_preprocessed)
print(X_test_preprocessed)

# Now, X_train_preprocessed and X_test_preprocessed are ready for model training and evaluation.
# Assuming X_train_preprocessed, X_test_preprocessed, y_train, y_test are already defined

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline with the preprocessing and the classifier
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', rf_classifier)])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Predictions
y_pred = model_pipeline.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


