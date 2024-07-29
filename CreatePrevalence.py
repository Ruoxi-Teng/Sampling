import numpy as np
import pandas as pd

# Define the prevalence values
prevalence_values = np.arange(0, 1.002, 0.001)

# Create an empty list to store the data
data = []

# First iteration: repeat each value 115 times
for prevalence in prevalence_values:
    data.extend([prevalence] * 115)

# Second iteration: repeat each value 230 times
for prevalence in prevalence_values:
    data.extend([prevalence] * 230)

# Create a DataFrame
df = pd.DataFrame({'Prevalence': data})

# Save to CSV
df.to_csv('prevalence_data.csv', index=False)

# Print some information about the generated data
print(f"Total rows in CSV: {len(df)}")
print(f"Number of unique prevalence values: {df['Prevalence'].nunique()}")
print(f"First few rows:")
print(df.head(10))
print("\nRows around the transition point:")
transition_point = len(prevalence_values) * 115
print(df.iloc[transition_point-5:transition_point+5])
print("\nLast few rows:")
print(df.tail(10))