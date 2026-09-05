import pandas as pd

DATA_PATH = "data/smart_manufacturing_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== UNIQUE VALUES =====")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique()[:20])