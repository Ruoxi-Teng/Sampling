import numpy as np
import pandas as pd

# Define the prevalence values
prevalence_values = np.arange(0, 1.000, 0.002)

# Create an empty list to store the data
data = []

# Generate the data
for prevalence in prevalence_values:
    # Repeat each prevalence value 230 times
    data.extend([prevalence] * 230)

# Create a DataFrame
df = pd.DataFrame({'Prevalence': data})

# Save to CSV
df.to_csv('prevalence_data.csv', index=False)

# Print some information about the generated data
print(f"Total rows in CSV: {len(df)}")
print(f"Number of unique prevalence values: {df['Prevalence'].nunique()}")
print(f"First few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())