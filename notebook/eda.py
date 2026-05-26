# IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# LOAD DATASET

df = pd.read_csv("../data/loan_prediction.csv")

# REMOVE EXTRA SPACES

df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully")

print(df.head())

# BASIC INFO

print("\nShape of Dataset:")

print(df.shape)

print("\nColumns:")

print(df.columns)

print("\nInfo:")

df.info()

print("\nDescription:")

print(df.describe())

# CHECK MISSING VALUES

print("\nMissing Values:")

print(df.isnull().sum())

# HANDLE MISSING VALUES

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:

    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:

    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Handling:")

print(df.isnull().sum().sum())

# CHECK DUPLICATES

print("\nDuplicate Rows:")

print(df.duplicated().sum())

# REMOVE DUPLICATES

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")

print(df.shape)

# LABEL ENCODING

encoder = LabelEncoder()

for col in cat_cols:

    df[col] = encoder.fit_transform(df[col])

print("\nCategorical Encoding Completed")

# SAVE CLEANED DATASET

df.to_csv(
    "../data/cleaned_loan_prediction.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully")