import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\VAISHNAVI\Downloads\diabetes.csv")

# Show first 5 rows
print("Original Data:")
print(df.head())

# Columns where 0 is not valid, so replace with NaN
cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero_invalid:
    df[col] = df[col].replace(0, np.nan)

# Fill missing values with median
for col in cols_with_zero_invalid:
    df[col] = df[col].fillna(df[col].median())

# Optional: remove outliers using IQR method
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

numeric_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

df = remove_outliers_iqr(df, numeric_cols)

# Save preprocessed dataset
df.to_csv("preprocessed_diabetes.csv", index=False)

print("\nPreprocessing completed successfully.")
print("Saved as: preprocessed_diabetes.csv")
print("Shape after preprocessing:", df.shape)