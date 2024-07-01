import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# we use the entire dataset for this strategy

df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(df.columns[[0]],axis=1)
df['CipRsum_prevalence'] = df['CipRsum']/df['TOTAL']
dt=pd.read_csv("Data/CIP_summary.csv")
dt=dt.drop(dt.columns[[0]],axis=1)
dt['CipRsum_prevalence'] = dt['CipRsum']/dt['TOTAL']


# check what we get if we use all the sites each year for sampling
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

prevalence_values = np.arange(0, 1.000, 0.002)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)

treatment_stats_sum['Sample'] = 'Total'


def select_random_clinics(df, num_clinics):
    unique_values = df['CLINIC'].unique()
    selected_values = np.random.choice(unique_values, num_clinics, replace=False)
    return df[df['CLINIC'].isin(selected_values)]


def generate_multiple_random_samples(df, num_clinics, num_samples, prevalence_values):
    np.random.seed(573)
    all_results = []

    for i in range(num_samples):
        subset_df = select_random_clinics(df, num_clinics).drop(columns=['CLINIC'])
        subset_df_agg = subset_df.groupby('YEAR').sum().reset_index()

        for col in subset_df_agg.columns[2:]:
            subset_df_agg[f'{col}_prevalence'] = subset_df_agg[col] / subset_df_agg['TOTAL']

        dt5 = dt.copy()
        dt5['CipRsum_prevalence'] = subset_df_agg['CipRsum_prevalence'].values

        treatment_stats = calculate_treatment_stats_sum(dt5, prevalence_values)
        treatment_stats['Sample'] = i + 1
        all_results.append(treatment_stats)

    results = pd.concat(all_results, ignore_index=True)
    return {'prevalence_data': subset_df_agg, 'treatment_stats': results}


def plot_treatment_paradigms(results, total_stats, num_clinics):
    plt.figure(figsize=(12, 6))
    for sample in results['Sample'].unique():
        sample_data = results[results['Sample'] == sample]
        plt.plot(sample_data['FailureToTreatPercentage'], sample_data['UnnecessaryUsePercentage'], color='gold',
                 alpha=0.3)

    mean_results = results.groupby('Prevalence')[['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'], color='red',
             label='Mean of samples')

    plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label='All sites')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Treatment paradigms under different cutoff value (2000-2022): Random {num_clinics} sites samples')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)

    return plt.gcf()


# using 5 sites
prevalence_values = np.arange(0, 1.000, 0.002)

# Generate samples
random_sample_results = generate_multiple_random_samples(df, num_clinics=5, num_samples=100, prevalence_values=prevalence_values)

# Calculate total stats
total_stats = calculate_treatment_stats_sum(dt, prevalence_values)

# Plot results
plot_5 = plot_treatment_paradigms(random_sample_results['treatment_stats'], total_stats, num_clinics=5)

plot_5.savefig('Figures/Random 5 sites.png')
# Display the plot
plt.show()

# Generate samples
random_sample_results_10 = generate_multiple_random_samples(df, num_clinics=10, num_samples=100, prevalence_values=prevalence_values)

# Calculate total stats
total_stats_10 = calculate_treatment_stats_sum(dt, prevalence_values)

# Plot results
plot_10 = plot_treatment_paradigms(random_sample_results_10['treatment_stats'], total_stats_10, num_clinics=10)

plot_10.savefig('Figures/Random 10 sites.png')
# Display the plot
plt.show()