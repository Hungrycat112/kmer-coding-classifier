import pandas as pd

# Load the CSV file
df = pd.read_csv("DNA.csv")

print("=== Dataset Info ===")
print(df.info())

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Class Distribution ===")
print(df['class'].value_counts())

print("\n=== Sequence Lengths ===")
df['length'] = df['sequence'].apply(len)
print(df['length'].describe())
