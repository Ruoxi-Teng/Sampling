import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz


def preprocess_data(data):
    min_year, max_year = data['YEAR'].min(), data['YEAR'].max()
    total_years = max_year - min_year + 1
    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index
    return data[data['CLINIC'].isin(sites_with_all_years)]


def calculate_treatment_stats_sum(data, prevalence_values):
    total = data['TOTAL'].sum()
    drug_change = (data['CipRsum_prevalence'].values[:, np.newaxis] >= prevalence_values).astype(int)
    failure_to_treat = ((1 - drug_change) * data['CipRsum'].values[:, np.newaxis]).sum(axis=0)
    unnecessary_use = (drug_change * (data['TOTAL'].values[:, np.newaxis] - data['CipRsum'].values[:, np.newaxis])).sum(
        axis=0)

    return pd.DataFrame({
        'Prevalence': prevalence_values,
        'FailureToTreat': failure_to_treat,
        'UnnecessaryUse': unnecessary_use,
        'Total': total,
        'UnnecessaryUsePercentage': unnecessary_use / total,
        'FailureToTreatPercentage': failure_to_treat / total
    })


def select_random_clinics(df, num_clinics, seed):
    np.random.seed(seed)
    unique_values = df['CLINIC'].unique()
    return df[df['CLINIC'].isin(np.random.choice(unique_values, num_clinics, replace=False))]


def generate_multiple_random_samples(df, num_clinics, num_samples, prevalence_values):
    all_results = []
    for i in range(num_samples):
        subset_df = select_random_clinics(df, num_clinics, 573 + i).drop(columns=['CLINIC'])
        subset_df_agg = subset_df.groupby('YEAR').sum().reset_index()
        subset_df_agg['CipRsum_prevalence'] = subset_df_agg['CipRsum'] / subset_df_agg['TOTAL']
        dt5 = dt.copy()
        dt5['CipRsum_prevalence'] = subset_df_agg['CipRsum_prevalence'].values
        treatment_stats = calculate_treatment_stats_sum(dt5, prevalence_values)
        treatment_stats['Sample'] = i + 1
        all_results.append(treatment_stats)
    return pd.concat(all_results, ignore_index=True)


def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


def plot_treatment_paradigms(results, total_stats, num_clinics):
    plt.figure(figsize=(12, 6))
    for sample in results['Sample'].unique():
        sample_data = results[results['Sample'] == sample]
        plt.plot(sample_data['FailureToTreatPercentage'], sample_data['UnnecessaryUsePercentage'], color='gold',
                 alpha=0.3)

    mean_results = results.groupby('Prevalence')[['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
    plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'], color='red',
             label=f'Mean of samples (AUC: {mean_auc:.4f})')

    total_auc = calculate_auc(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'])
    plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Treatment paradigms under different cutoff value (2000-2022): Random {num_clinics} sites samples')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf()


# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df=df.drop(columns=df.columns[0])
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt=dt.drop(columns=dt.columns[0])

df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipRsum_prevalence'] = dt1['CipRsum'] / dt1['TOTAL']

df1 = preprocess_data(df)
prevalence_values = np.arange(0, 1.000, 0.002)
total_stats = calculate_treatment_stats_sum(dt, prevalence_values)

# Generate and plot results for 5 and 10 clinics
for num_clinics in [5, 10]:
    random_sample_results = generate_multiple_random_samples(df1, num_clinics, 100, prevalence_values)
    random_sample_results.to_csv(f"Data/random_summary_{num_clinics}.csv", index=False)
    mean_results = random_sample_results.groupby('Prevalence')[
        ['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_results.to_csv(f"Data/random_summary_mean_{num_clinics}.csv", index=False)

    plot = plot_treatment_paradigms(random_sample_results, total_stats, num_clinics)
    plot.savefig(f'Figures/Random {num_clinics} sites.png')
    plt.show()