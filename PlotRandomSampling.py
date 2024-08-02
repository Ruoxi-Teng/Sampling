import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy import interpolate

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
        np.random.seed(573 + i)
        yearly_data = []

        for year in df['YEAR'].unique():
            year_df = df[df['YEAR'] == year]
            selected_clinics = np.random.choice(year_df['CLINIC'].unique(), num_clinics, replace=False)
            subset_df = year_df[year_df['CLINIC'].isin(selected_clinics)]
            yearly_data.append(subset_df)

            # Store selected clinics information
            selected_clinics_data.append({
                'Sample': i + 1,
                'Year': year,
                'Selected_Clinics': ', '.join(selected_clinics)
            })

        subset_df_agg = pd.concat(yearly_data).groupby('YEAR').sum().reset_index()
        subset_df_agg['CipRsum_prevalence'] = subset_df_agg['CipRsum'] / subset_df_agg['TOTAL']

        dt5 = dt.copy()
        dt5['CipRsum_prevalence'] = subset_df_agg['CipRsum_prevalence'].values
        treatment_stats = calculate_treatment_stats_sum(dt5, prevalence_values)
        treatment_stats['Sample'] = i + 1
        all_results.append(treatment_stats)

    # Create a dataframe with selected clinics information
    selected_clinics_df = pd.DataFrame(selected_clinics_data)

    return pd.concat(all_results, ignore_index=True), selected_clinics_df


def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


def plot_treatment_paradigms(results, total_stats, num_clinics):
    plt.figure(figsize=(12, 6),dpi=700)

    # Define common x-axis (FailureToTreatPercentage) for interpolation
    x_common = np.linspace(results['FailureToTreatPercentage'].min(),
                           results['FailureToTreatPercentage'].max(),
                           1000)

    interpolated_data = []

    for sample in results['Sample'].unique():
        sample_data = results[results['Sample'] == sample]
        # Remove duplicates
        sample_data = sample_data.drop_duplicates(subset='FailureToTreatPercentage')

        # Sort the data by FailureToTreatPercentage
        sample_data = sample_data.sort_values('FailureToTreatPercentage')
        # Interpolate
        f = interpolate.interp1d(sample_data['FailureToTreatPercentage'],
                                 sample_data['UnnecessaryUsePercentage'],
                                 kind='linear', fill_value='extrapolate')
        y_interp = f(x_common)

        interpolated_data.append(y_interp)

        plt.plot(x_common, y_interp, color='gold', alpha=0.01, linewidth=0.7)

        # Calculate mean of interpolated data
    y_mean = np.mean(interpolated_data, axis=0)

    # Calculate AUC for mean curve
    mean_auc = calculate_auc(x_common, y_mean)

    plt.plot(x_common, y_mean, color='red',
             label=f'Mean of samples (AUC: {mean_auc:.4f})')
    total_auc = calculate_auc(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'])
    plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    # plt.yscale('exp')
    plt.title(f'Treatment paradigms under different cutoff value (2000-2022): Random {num_clinics} sites samples')
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

df1 = preprocess_data(df)
prevalence_values = np.arange(0, 1.000, 0.002)
total_stats = calculate_treatment_stats_sum(dt, prevalence_values)

# Generate and plot results for 5 and 10 clinics
for num_clinics in [1, 5, 10]:
    random_sample_results, selected_clinics_df = generate_multiple_random_samples(df1, num_clinics, 1500,
                                                                                  prevalence_values)
    random_sample_results.to_csv(f"Data/random_summary_{num_clinics}.csv", index=False)
    mean_results = random_sample_results.groupby('Prevalence')[
        ['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_results.to_csv(f"Data/random_summary_mean_{num_clinics}.csv", index=False)

    # Print the first few rows of the selected clinics dataframe for checking
    print(f"\nSelected clinics for {num_clinics} clinics (first 10 rows):")
    print(selected_clinics_df.head(10))

    plot = plot_treatment_paradigms(random_sample_results, total_stats, num_clinics)
    plot.savefig(f'Figures/Random {num_clinics} sites.png')
    plt.show()