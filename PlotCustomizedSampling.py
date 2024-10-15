import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(columns=df.columns[0])
df['CipR_Sum_prevalence'] = df['CipR_Sum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(columns=dt.columns[0])

def preprocess_data(data):
    min_year, max_year = data['YEAR'].min(), data['YEAR'].max()
    total_years = max_year - min_year + 1
    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index
    return data[data['CLINIC'].isin(sites_with_all_years)]


df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipR_Sum_prevalence'] = dt1['CipR_Sum'] / dt1['TOTAL']

#calculate treatment failure and unnecessary treatment
def calculate_treatment_stats_sum(data, prevalence_values):
    total = data['TOTAL'].sum()

    # Initialize arrays to store results
    failure_to_treat = np.zeros(len(prevalence_values))
    unnecessary_use = np.zeros(len(prevalence_values))

    for i, prevalence_value in enumerate(prevalence_values):
        drug_changed = False
        for _, row in data.iterrows():
            if not drug_changed and row['CipR_Sum_prevalence'] >= prevalence_value:
                drug_changed = True

            if drug_changed:
                unnecessary_use[i] += row['TOTAL'] - row['CipR_Sum']
            else:
                failure_to_treat[i] += row['CipR_Sum']

    return pd.DataFrame({
        'Prevalence': prevalence_values,
        'FailureToTreat': failure_to_treat,
        'UnnecessaryUse': unnecessary_use,
        'Total': total,
        'UnnecessaryUsePercentage': unnecessary_use / total,
        'FailureToTreatPercentage': failure_to_treat / total
        })


# fixed sampling approach
def calculate_treatment_stats_sites(df, prevalence_values):
    # Ensure the DataFrame is sorted by YEAR
    df = df.sort_values('YEAR')
    clinic_total = []
    # Get unique years and clinics
    years = df['YEAR'].unique()
    all_clinics = df['CLINIC'].unique()

    for clinic in all_clinics:
        # Filter data for the current year and selected clinics
        clinic_data = df[(df['CLINIC'] == clinic)]
        # calculate for each sites each year the unnecessary use and failure to treat under each prevalence
        clinic_data_stats = calculate_treatment_stats_sum(clinic_data, prevalence_values)
        clinic_data_stats['CLINIC'] = clinic
        clinic_total.append(clinic_data_stats)
    # append the result of each site together
    clinic_total = pd.concat(clinic_total, ignore_index=True)
    clinic_total.columns = ['Prevalence', 'FailureToTreat', 'UnnecessaryUse', 'Total',
                            'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'CLINIC']

    # sum the failure to treat and unnecessary under each prevalence
    clinic_sum = clinic_total.groupby('Prevalence')[['FailureToTreat', 'UnnecessaryUse']].sum().reset_index()
    clinic_sum['TOTAL'] = df['TOTAL'].sum()
    # recalculate the failure to treat and unnecessary treatment percentage
    clinic_sum['UnnecessaryUsePercentage'] = clinic_sum['UnnecessaryUse'] / clinic_sum['TOTAL']
    clinic_sum['FailureToTreatPercentage'] = clinic_sum['FailureToTreat'] / clinic_sum['TOTAL']

    return clinic_sum

def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))




prevalence_values = np.arange(0, 1.02, 0.02)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
treatment_stats_customized = calculate_treatment_stats_sites(df, prevalence_values)


total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'])
customized_auc = calculate_auc(treatment_stats_customized['FailureToTreatPercentage'], treatment_stats_customized['UnnecessaryUsePercentage'])
plt.figure(figsize=(12, 6),dpi=700)
plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites National (AUC: {total_auc:.4f})')
plt.plot(treatment_stats_customized['FailureToTreatPercentage'], treatment_stats_customized['UnnecessaryUsePercentage'], color='red',
             label=f'All sites Customized (AUC: {customized_auc:.4f})')
plt.xlabel('Failure to Treat (%)')
plt.ylabel('Unnecessary Treatment (%)')
plt.title(f'Failure to Treat v.s. Unnecessary Treatment under different threshold (2000-2022): Customized v.s. All sites')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.legend()
plt.grid(True)
plt.savefig('Figures/Customized Sampling_GISP.png')
plt.show()