import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

df = pd.read_csv("Data/CIP_summary.csv")
df = df.drop(df.columns[[0]], axis=1)
dt = pd.read_csv("Data/CIP_summary_by_sites.csv")
dt = dt.drop(dt.columns[[0]], axis=1)
dt['CipRsum_prevalence'] = dt['CipRsum'] / dt['TOTAL']

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

def calculate_treatment_stats_new(subset_dt, prevalence_values):
    results_list = []

    for prevalence_value in prevalence_values:
        tx_stats = subset_dt.copy()

        tx_stats['DrugChange'] = (tx_stats['CipRsum_prevalence'] >= prevalence_value).astype(int)
        tx_stats['FailureToTreat'] = tx_stats.apply(lambda row: 0 if row['DrugChange'] == 1 else row['CipRsum'], axis=1)
        tx_stats['UnnecessaryUse'] = tx_stats.apply(
            lambda row: row['TOTAL'] - row['CipRsum'] if row['DrugChange'] == 1 else 0, axis=1)

        tx_stats_selected = tx_stats[['YEAR', 'TOTAL', 'FailureToTreat', 'UnnecessaryUse']]

        tx_stats_selected = tx_stats_selected.groupby('YEAR').sum().reset_index()

        results = {
            'FailureToTreat': tx_stats_selected['FailureToTreat'].sum(),
            'UnnecessaryUse': tx_stats_selected['UnnecessaryUse'].sum(),
            'Total': tx_stats['TOTAL'].sum(),
            'CutoffValue': prevalence_value
        }
        results['UnnecessaryUsePercentage'] = results['UnnecessaryUse'] / results['Total']
        results['FailureToTreatPercentage'] = results['FailureToTreat'] / results['Total']

        results_list.append(results)

    return pd.DataFrame(results_list)


def calculate_auc(x, y):
    # Ensure the data is sorted by x values
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)

    # Calculate AUC
    auc = trapz(y_sorted, x_sorted)

    # Normalize AUC to be between 0 and 1
    auc_normalized = auc / (max(x_sorted) * max(y_sorted))

    return auc_normalized

prevalence_values = np.arange(0, 1.000, 0.002)
treatment_stats_sum = calculate_treatment_stats_sum(df, prevalence_values)
results_df = calculate_treatment_stats_new(dt, prevalence_values)

treatment_stats_sum['Sample'] = 'Uniform'
results_df['Sample'] = 'Customized'
results_df = results_df.rename(columns={'CutoffValue': 'Prevalence'})

plot_customized = pd.concat([
    treatment_stats_sum[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Sample']],
    results_df[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Sample']]
])
plot_customized = plot_customized.rename(columns={'Sample': 'Method'})

plt.figure(figsize=(12, 6))

for method in plot_customized['Method'].unique():
    data = plot_customized[plot_customized['Method'] == method].sort_values('FailureToTreatPercentage')
    auc = calculate_auc(data['FailureToTreatPercentage'], data['UnnecessaryUsePercentage'])
    plt.plot(data['FailureToTreatPercentage'], data['UnnecessaryUsePercentage'],
             label=f"{method} (AUC: {auc:.4f})")

plt.xlabel('Failure to Treat (%)')
plt.ylabel('Unnecessary Treatment (%)')
plt.title('Customized V.S. Uniform')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Figures/Customized v.s Uniform.png')
plt.show()

# Calculate and print AUC values
auc_results = []
for method in plot_customized['Method'].unique():
    data = plot_customized[plot_customized['Method'] == method].sort_values('FailureToTreatPercentage')
    auc = calculate_auc(data['FailureToTreatPercentage'], data['UnnecessaryUsePercentage'])
    auc_results.append({'Method': method, 'AUC': auc})

auc_df = pd.DataFrame(auc_results)
print("AUC Results:")
print(auc_df)

# Save AUC results to CSV
auc_df.to_csv('Data/auc_results_customized_vs_uniform.csv', index=False)