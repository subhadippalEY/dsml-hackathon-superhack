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
df = pd.read_csv('C:\\Users\XZ957MG\\source\\repos\\IIIMRoorkie-Assignment\\IIIMRoorkie-Assignment\\Hackathon\\train_data.csv')
df_test = pd.read_csv('C:\\Users\XZ957MG\\source\\repos\\IIIMRoorkie-Assignment\\IIIMRoorkie-Assignment\\Hackathon\\test_data.csv')

# Splitting the dataset into features and target variable
X = df.drop(['id', 'defects'], axis=1)
y = df['defects']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size = 0.2, random_state=42) #test_size=32056, train_size=74794, random_state=42)


# Splitting the dataset into features and target variable
#X_train = df.drop(['id', 'defects'], axis=1)
#y_train = df['defects']

# Handle missing values
#X_train = X_train.fillna(0)

# Splitting the dataset into features and target variable
X_test = df_test.drop(['id'], axis=1)
# Then, add a new column "defects" with all values set to 0
X_test['defects'] = 0

y_test = X_test['defects']
X_test = X_test.drop(['defects'], axis=1) 

# Handle missing values
#X_test = X_test.fillna(0)

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


import matplotlib.pyplot as plt
import seaborn as sns


# # Setting the aesthetic style of the plots
# sns.set_style("whitegrid")

# # Distribution of defects
# plt.figure(figsize=(6, 4))
# sns.countplot(x='defects', data=df)
# plt.title('Distribution of Defects')
# plt.show()

# # Correlation matrix heatmap
# plt.figure(figsize=(12, 10))
# corr_matrix = df[numerical_cols].corr()
# sns.heatmap(corr_matrix, annot=True, fmt=".2f")
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()

# # Boxplot for McCabe Cyclomatic Complexity
# plt.figure(figsize=(6, 4))
# sns.boxplot(x='defects', y='McCabeCyclomaticComplexity', data=df)
# plt.title('McCabe Cyclomatic Complexity by Defects')
# plt.show()

# # Distribution of Code Age
# plt.figure(figsize=(6, 4))
# sns.histplot(df['CodeAge'], bins=30, kde=True)
# plt.title('Distribution of Code Age')
# plt.show()

# # Exploring categorical data: CodeLanguage distribution
# plt.figure(figsize=(8, 6))
# sns.countplot(y='CodeLanguage', data=df, order = df['CodeLanguage'].value_counts().index)
# plt.title('Distribution of Code Languages')
# plt.show()


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




