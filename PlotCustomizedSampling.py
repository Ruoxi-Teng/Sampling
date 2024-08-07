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
    total = data['TOTAL'].sum()

    # Initialize arrays to store results
    failure_to_treat = np.zeros(len(prevalence_values))
    unnecessary_use = np.zeros(len(prevalence_values))

    for i, prevalence_value in enumerate(prevalence_values):
        drug_changed = False
        for _, row in data.iterrows():
            if not drug_changed and row['CipRsum_prevalence'] >= prevalence_value:
                drug_changed = True

            if drug_changed:
                unnecessary_use[i] += row['TOTAL'] - row['CipRsum']
            else:
                failure_to_treat[i] += row['CipRsum']

    return pd.DataFrame({
        'Prevalence': prevalence_values,
        'FailureToTreat': failure_to_treat,
        'UnnecessaryUse': unnecessary_use,
        'Total': total,
        'UnnecessaryUsePercentage': unnecessary_use / total,
        'FailureToTreatPercentage': failure_to_treat / total
        })


def calculate_treatment_stats_new(subset_dt, prevalence_values):
    results_list = []

    for prevalence_value in prevalence_values:
        tx_stats = subset_dt.copy()

        # Initialize DrugChange column
        tx_stats['DrugChange'] = False

        # Apply the drug change logic for each site
        for site in tx_stats['CLINIC'].unique():
            site_data = tx_stats[tx_stats['CLINIC'] == site]
            switch_year = site_data[site_data['CipRsum_prevalence'] >= prevalence_value]['YEAR'].min()

            if pd.notnull(switch_year):
                tx_stats.loc[(tx_stats['CLINIC'] == site) & (tx_stats['YEAR'] >= switch_year), 'DrugChange'] = True

        # Calculate FailureToTreat and UnnecessaryUse
        tx_stats['FailureToTreat'] = tx_stats.apply(lambda row: 0 if row['DrugChange'] else row['CipRsum'], axis=1)
        tx_stats['UnnecessaryUse'] = tx_stats.apply(
            lambda row: row['TOTAL'] - row['CipRsum'] if row['DrugChange'] else 0, axis=1)

        results = {
            'FailureToTreat': tx_stats['FailureToTreat'].sum(),
            'UnnecessaryUse': tx_stats['UnnecessaryUse'].sum(),
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

prevalence_values = np.arange(0, 1.02, 0.02)
treatment_stats_sum = calculate_treatment_stats_sum(df, prevalence_values)
results_df = calculate_treatment_stats_new(dt, prevalence_values)
print(treatment_stats_sum)
plt.figure(figsize=(12, 6))
total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'])
plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

customized_auc = calculate_auc(results_df['FailureToTreatPercentage'], results_df['UnnecessaryUsePercentage'])
plt.plot(results_df['FailureToTreatPercentage'], results_df['UnnecessaryUsePercentage'], color='red',
             label=f'Customized All sites (AUC: {customized_auc:.4f})')

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
