import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz

# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(df.columns[[0]], axis=1)
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(dt.columns[[0]], axis=1)

data = pd.read_csv("Data/All_Selected_Sites_Sample.csv")
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

def calculate_estimated_prevalence(data):
    # Calculate estimated prevalence for each year, threshold, and number of sites
    prevalence_df = data.groupby(['YEAR', 'Threshold', 'NumSites']).apply(
        lambda x: pd.Series({
            'EstimatedPrevalence': x['CipRsum'].sum() / x['TOTAL'].sum(),
            'TotalSamples': x['TOTAL'].sum(),
            'TotalPositive': x['CipRsum'].sum()
        })
    ).reset_index()

    # Sort the DataFrame by year and threshold
    prevalence_df = prevalence_df.sort_values(['Threshold','NumSites'])

    return prevalence_df

def calculate_stats_sample(prevalence_df):
    # Get unique combinations of prevalence values and NumSites
    prevalence_numsites = prevalence_df[['Threshold', 'NumSites']].drop_duplicates()

    # Initialize the result list
    result = []

    for _, row in prevalence_numsites.iterrows():
        prevalence_value = row['Threshold']
        num_sites = row['NumSites']

        # Filter data for this prevalence value and NumSites
        data = prevalence_df[(prevalence_df['Threshold'] == prevalence_value) &
                             (prevalence_df['NumSites'] == num_sites)].sort_values('YEAR')

        total = data['TotalSamples'].sum()
        drug_changed = False
        failure_to_treat = 0
        unnecessary_use = 0

        for _, data_row in data.iterrows():
            if not drug_changed and data_row['EstimatedPrevalence'] >= prevalence_value:
                drug_changed = True

            if drug_changed:
                unnecessary_use += data_row['TotalSamples'] - data_row['TotalPositive']
            else:
                failure_to_treat += data_row['TotalPositive']

        result.append({
            'Prevalence': prevalence_value,
            'NumSites': num_sites,
            'FailureToTreat': failure_to_treat,
            'UnnecessaryUse': unnecessary_use,
            'Total': total,
            'UnnecessaryUsePercentage': unnecessary_use / total if total > 0 else 0,
            'FailureToTreatPercentage': failure_to_treat / total if total > 0 else 0
        })

    return pd.DataFrame(result)


def calculate_stats_pop(prevalence_df):
    # Get unique combinations of prevalence values and NumSites
    prevalence_numsites = prevalence_df[['Threshold', 'NumSites']].drop_duplicates()

    # Initialize the result list
    result = []

    for _, row in prevalence_numsites.iterrows():
        prevalence_value = row['Threshold']
        num_sites = row['NumSites']

        # Filter data for this prevalence value and NumSites
        data = prevalence_df[(prevalence_df['Threshold'] == prevalence_value) &
                             (prevalence_df['NumSites'] == num_sites)].sort_values('YEAR')

        data.reset_index(drop=True, inplace=True)
        dt.reset_index(drop=True, inplace=True)
        data = pd.concat([data,dt[['TOTAL','CipRsum']]],axis=1)


        total = data['TOTAL'].sum()
        drug_changed = False
        failure_to_treat = 0
        unnecessary_use = 0

        for _, data_row in data.iterrows():
            if not drug_changed and data_row['EstimatedPrevalence'] >= prevalence_value:
                drug_changed = True

            if drug_changed:
                unnecessary_use += data_row['TOTAL'] - data_row['CipRsum']
            else:
                failure_to_treat += data_row['CipRsum']

        result.append({
            'Prevalence': prevalence_value,
            'NumSites': num_sites,
            'FailureToTreat': failure_to_treat,
            'UnnecessaryUse': unnecessary_use,
            'Total': total,
            'UnnecessaryUsePercentage': unnecessary_use / total if total > 0 else 0,
            'FailureToTreatPercentage': failure_to_treat / total if total > 0 else 0
        })

    return pd.DataFrame(result)

def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


prevalence_df=calculate_estimated_prevalence(data)

result=calculate_stats_pop(prevalence_df)
result=result.sort_values(['NumSites','Prevalence'],ignore_index=True)
result_5=result[result['NumSites']==5]
result_5=result_5.sort_values(['FailureToTreatPercentage','UnnecessaryUsePercentage'],ignore_index=True)
result_10=result[result['NumSites']==10]
result_10=result_10.sort_values(['FailureToTreatPercentage','UnnecessaryUsePercentage'],ignore_index=True)

prevalence_values = np.arange(0, 1.02, 0.02)
total_stats = calculate_treatment_stats_sum(dt, prevalence_values)
total_auc = calculate_auc(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'])
plt.figure(figsize=(12, 6))
plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

auc_5 = calculate_auc(result_5['FailureToTreatPercentage'], result_5['UnnecessaryUsePercentage'])
plt.plot(result_5['FailureToTreatPercentage'], result_5['UnnecessaryUsePercentage'], color='orange',
             label=f'5 sites (AUC: {auc_5:.4f})')

auc_10 = calculate_auc(result_10['FailureToTreatPercentage'], result_10['UnnecessaryUsePercentage'])
plt.plot(result_10['FailureToTreatPercentage'], result_10['UnnecessaryUsePercentage'], color='green',
             label=f'10 sites (AUC: {auc_10:.4f})')

plt.xlabel('Failure to Treat (%)')
plt.ylabel('Unnecessary Treatment (%)')
plt.title(f'Min Cost Strategy for total population')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/Min Cost Pop.png')
plt.show()


result_s=calculate_stats_sample(prevalence_df)
result_s=result_s.sort_values(['NumSites','Prevalence'],ignore_index=True)
result_5s=result_s[result_s['NumSites']==5]
result_5s=result_5s.sort_values(['FailureToTreatPercentage','UnnecessaryUsePercentage'],ignore_index=True)
result_10s=result_s[result_s['NumSites']==10]
result_10s=result_10s.sort_values(['FailureToTreatPercentage','UnnecessaryUsePercentage'],ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

auc_5s = calculate_auc(result_5s['FailureToTreatPercentage'], result_5s['UnnecessaryUsePercentage'])
plt.plot(result_5s['FailureToTreatPercentage'], result_5s['UnnecessaryUsePercentage'], color='orange',
             label=f'5 sites (AUC: {auc_5s:.4f})')

auc_10s = calculate_auc(result_10s['FailureToTreatPercentage'], result_10s['UnnecessaryUsePercentage'])
plt.plot(result_10s['FailureToTreatPercentage'], result_10s['UnnecessaryUsePercentage'], color='green',
             label=f'10 sites (AUC: {auc_10s:.4f})')

plt.xlabel('Failure to Treat (%)')
plt.ylabel('Unnecessary Treatment (%)')
plt.title(f'Min Cost Strategy for sampled population')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/Min Cost Sample.png')
plt.show()