import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(df.columns[[0]], axis=1)
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(dt.columns[[0]], axis=1)

def calculate_treatment_stats_sum(data, prevalence_values):
    results = []
    total = data['TOTAL'].sum()

    for prevalence in prevalence_values:
        drug_change = (data['CipRsum_prevalence'] >= prevalence).astype(int)
        failure_to_treat = ((1 - drug_change) * data['CipRsum']).sum()
        unnecessary_use = (drug_change * (data['TOTAL'] - data['CipRsum'])).sum()

        results.append({
            'Prevalence': prevalence,
            'FailureToTreat': failure_to_treat,
            'UnnecessaryUse': unnecessary_use,
            'Total': total,
            'UnnecessaryUsePercentage': unnecessary_use / total,
            'FailureToTreatPercentage': failure_to_treat / total
        })
    return pd.DataFrame(results)

def preprocess_data(data):
    min_year = data['YEAR'].min()
    max_year = data['YEAR'].max()
    total_years = max_year - min_year + 1

    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index

    return data[data['CLINIC'].isin(sites_with_all_years)]

df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipRsum_prevalence'] = dt1['CipRsum'] / dt1['TOTAL']

def calculate_lambda(data, threshold, epsilon=1e-10):
    data_sorted = data.sort_values('FailureToTreatPercentage')
    idx = np.searchsorted(data_sorted['FailureToTreatPercentage'], threshold)

    if idx == 0:
        idx = 1
    elif idx == len(data_sorted):
        idx = len(data_sorted) - 1

    point1 = data_sorted.iloc[idx - 1]
    point2 = data_sorted.iloc[idx]

    delta_failure = point2['FailureToTreatPercentage'] - point1['FailureToTreatPercentage']
    delta_unnecessary = point2['UnnecessaryUsePercentage'] - point1['UnnecessaryUsePercentage']

    if abs(delta_failure) < epsilon:
        slope = np.inf if delta_unnecessary > epsilon else -np.inf if delta_unnecessary < -epsilon else 0
    else:
        slope = delta_unnecessary / delta_failure

    lambda_val = -1 / slope if abs(slope) > epsilon else np.inf

    return lambda_val

prevalence_values = np.arange(0, 1.002, 0.002)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)

# Calculate lambda for all thresholds
lambda_results = []
for threshold in prevalence_values:
    lambda_val = calculate_lambda(treatment_stats_sum, threshold)
    lambda_results.append({'Threshold': threshold, 'Lambda': lambda_val})

# Create and save lambda results
lambda_df = pd.DataFrame(lambda_results)
print(lambda_df)
lambda_df.to_csv('Data/lambda_results.csv', index=False)

# Optional: Plot lambda values
plt.figure(figsize=(12, 6))
plt.plot(lambda_df['Threshold'], lambda_df['Lambda'])
plt.title('Lambda Values for Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Lambda')
plt.yscale('symlog')  # Use symlog scale to handle both positive and negative values
plt.grid(True)
plt.savefig('Figures/lambda_values.png')
plt.show()