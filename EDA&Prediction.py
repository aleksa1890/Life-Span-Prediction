import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# Load CSV file into data frame
df = pd.read_csv('LifeExpectancyData.csv')
df.info()

# Check for missing values and calculate their proportions
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / len(df)) * 100

# Create a DataFrame to summarize missing values
missing_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})

# Display columns with missing values only
missing_summary[missing_summary['Missing Values'] > 0]

df = df.dropna()

# Outlier detection
# Define a function for outlier removal
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 9.5 * IQR
        upper_bound = Q3 + 9.5 * IQR
        # Remove rows with outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Select numerical columns for outlier detection
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Remove outliers
data_cleaned = remove_outliers(df, numerical_columns)

# Check the shape of the dataset after removing outliers
data_cleaned.shape

# Encode categorical variables
label_encoder = LabelEncoder()
data_cleaned['Status'] = label_encoder.fit_transform(data_cleaned['Status'])

# Optionally, drop 'Country' if it's not relevant for modeling
data_cleaned = data_cleaned.drop('Country', axis=1)

# Standardize numerical columns
scaler = StandardScaler()
numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns

data_cleaned[numerical_columns] = scaler.fit_transform(data_cleaned[numerical_columns])

# Display the transformed data
data_cleaned.head()

# Define the target variable (Life expectancy) and features
X = data_cleaned.drop('Life expectancy ', axis=1)
y = data_cleaned['Life expectancy ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the splits
X_train.shape, X_test.shape, y_train.shape, y_test.shape