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


def generate_multiple_random_samples(df, num_clinics, num_samples, prevalence_values):
    all_results = []
    selected_clinics_data = []

    for i in range(num_samples):
        np.random.seed(573 + i * num_clinics)
        unique_values = df['CLINIC'].unique()
        selected_clinics = np.random.choice(unique_values, num_clinics, replace=False)

        subset_df = df[df['CLINIC'].isin(selected_clinics)].drop(columns=['CLINIC'])
        subset_df_agg = subset_df.groupby('YEAR').sum().reset_index()
        subset_df_agg['CipRsum_prevalence'] = subset_df_agg['CipRsum'] / subset_df_agg['TOTAL']

        dt5 = dt.copy()
        dt5['CipRsum_prevalence'] = subset_df_agg['CipRsum_prevalence'].values
        treatment_stats = calculate_treatment_stats_sum(dt5, prevalence_values)
        treatment_stats['Sample'] = i + 1
        all_results.append(treatment_stats)

        # Store selected clinics information
        selected_clinics_data.append({
            'Sample': i + 1,
            'Selected_Clinics': ', '.join(selected_clinics)
        })

    # Create a dataframe with selected clinics information
    selected_clinics_df = pd.DataFrame(selected_clinics_data)

    return pd.concat(all_results, ignore_index=True), selected_clinics_df


def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


def plot_treatment_paradigms(results, total_stats, num_clinics):
    plt.figure(figsize=(12, 6))
    for sample in results['Sample'].unique():
        sample_data = results[results['Sample'] == sample]
        plt.scatter(sample_data['FailureToTreatPercentage'], sample_data['UnnecessaryUsePercentage'], color='grey', alpha=0.1, s=10)

    mean_results = results.groupby('Prevalence')[['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
    plt.scatter(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'], color='red', alpha=0.1, s=10,
             label=f'Mean of samples (AUC: {mean_auc:.4f})')

    total_auc = calculate_auc(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'])
    plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Treatment paradigms under different cutoff value (2000-2022): Fixed {num_clinics} sites samples')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf()


# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(columns=df.columns[0])
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(columns=dt.columns[0])

df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipRsum_prevalence'] = dt1['CipRsum'] / dt1['TOTAL']

prevalence_values = np.arange(0, 1.002, 0.002)
total_stats = calculate_treatment_stats_sum(dt, prevalence_values)

# Generate and plot results for 5 and 10 clinics
for num_clinics in [1, 5, 10]:
    random_sample_results, selected_clinics_df = generate_multiple_random_samples(df1, num_clinics, 1500,
                                                                                  prevalence_values)
    random_sample_results.to_csv(f"Data/fixed_summary_{num_clinics}.csv", index=False)
    mean_results = random_sample_results.groupby('Prevalence')[
        ['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_results.to_csv(f"Data/fixed_summary_mean_{num_clinics}.csv", index=False)

    # Print the first 10 rows of the selected clinics dataframe for checking
    print(f"\nSelected clinics for {num_clinics} clinics (first 10 rows):")
    print(selected_clinics_df.head(10))

    plot = plot_treatment_paradigms(random_sample_results, total_stats, num_clinics)
    plot.savefig(f'Figures/Fixed {num_clinics} sites dots.png')
    plt.show()