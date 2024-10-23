import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import seaborn as sns

df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(columns=df.columns[0])
df['CipR_Sum_prevalence'] = df['CipR_Sum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(columns=dt.columns[0])

# Load and preprocess data
# df = pd.read_csv("Data/site_prevalence_data.csv")
# df = df.drop(columns=df.columns[0])
# dt = pd.read_csv("Data/total_prevalence_data.csv")
# dt = dt.drop(columns=dt.columns[0])

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

# fixed sampling approach
def select_random_clinics_multiple_samples(df, num_clinics, num_samples):
    np.random.seed(10000)
    # Ensure the DataFrame is sorted by YEAR
    df = df.sort_values('YEAR')

    # Get unique years
    years = df['YEAR'].unique()

    # Initialize an empty list to store results from all samples
    all_samples_results = []

    for sample in range(1, num_samples + 1):
        # Initialize an empty list to store selected data for this sample
        selected_data = []

        for year in years:
            # Filter data for the current year
            year_data = df[df['YEAR'] == year]

            # If there are fewer clinics than requested, select all of them
            if len(year_data) <= num_clinics:
                selected_data.append(year_data)
            else:
                # Randomly select the specified number of clinics
                selected_clinics = np.random.choice(year_data['CLINIC'].unique(), num_clinics, replace=False)
                selected_year_data = year_data[year_data['CLINIC'].isin(selected_clinics)]
                selected_data.append(selected_year_data)

        # Concatenate all selected data into a single DataFrame for this sample
        result_df = pd.concat(selected_data, ignore_index=True)

        # Calculate estimated prevalence
        result_total = result_df.groupby(['YEAR'])[['TOTAL', 'CipR_Sum']].sum().reset_index()
        result_total['CipR_Sum_prevalence'] = result_total['CipR_Sum'] / result_total['TOTAL']

        # Merge back with original data to keep all variables
        result = pd.concat([dt.iloc[:, 0:3], result_total[['CipR_Sum_prevalence']]], axis=1)

        # Add sample indicator
        result['Sample'] = sample

        # Append to the list of all samples results
        all_samples_results.append(result)

    # Concatenate results from all samples
    final_result = pd.concat(all_samples_results, ignore_index=True)

    return final_result

def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


# plot fixed sampling result
# plot random sampling result
def plot_random_results(results, treatment_stats_sum, num_clinics):
    plt.figure(figsize=(12, 6), dpi=700)
    sample_total = []
    # plot each sample curve
    for sample in results['Sample'].unique():
        sample_data = results[results['Sample'] == sample]
        sample_final = calculate_treatment_stats_sum(sample_data, prevalence_values)
        sample_final['Sample'] = sample
        # Append the sample_final dataframe to the sample_total list
        sample_total.append(sample_final)
        plt.plot(sample_final['FailureToTreatPercentage'], sample_final['UnnecessaryUsePercentage'], color='gold')
    # Concatenate all sample_final dataframes into a single dataframe
    sample_total = pd.concat(sample_total, ignore_index=True)
    sample_total.columns = ['Prevalence', 'FailureToTreat', 'UnnecessaryUse', 'Total',
                            'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Sample']

    # plot mean result
    mean_results = sample_total.groupby('Prevalence')[['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
    plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'], color='red',
             label=f'Mean of samples (AUC: {mean_auc:.4f})')

    # plot all sites curve
    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(
        f'Failure to Treat v.s. Unnecessary Treatment under different threshold (2000-2022): Random {num_clinics} sites samples')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf()


prevalence_values = np.arange(0, 1.02, 0.02)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
treatment_stats_customized = calculate_treatment_stats_sites(df, prevalence_values)

# Generate and plot results for 5 and 10 clinics
# result: random sampling
# for num_clinics in [5, 10, 20]:
    # random_sample_results= select_random_clinics_multiple_samples(df1, num_clinics, 1500)
    # plot = plot_random_results(random_sample_results, treatment_stats_sum, num_clinics)
    # plot.savefig(f'Figures/Random {num_clinics} sites_Simulated.png')
    # plt.show()


def plot_combined_results(results_dict, treatment_stats_sum, treatment_stats_customized, prevalence_values):
    plt.figure(figsize=(12, 8), dpi=700)

    colors = ['orange', 'gold', 'green']

    # Plot mean results with confidence intervals for each number of clinics
    for i, (num_clinics, results) in enumerate(results_dict.items()):
        sample_total = []
        for sample in results['Sample'].unique():
            sample_data = results[results['Sample'] == sample]
            sample_final = calculate_treatment_stats_sum(sample_data, prevalence_values)
            sample_final['Sample'] = sample
            sample_total.append(sample_final)

        sample_total_df = pd.concat(sample_total, ignore_index=True)
        mean_results = sample_total_df.groupby('Prevalence')[
            ['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
        mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])

        plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'],
                 color=colors[i], label=f'Mean of {num_clinics} clinics (AUC: {mean_auc:.4f})')

        # Plot treatment_stats_sum
    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue', label=f'United All sites (AUC: {total_auc:.4f})')

    # Plot customized switch curve (treatment_stats_customized)
    customized_auc = calculate_auc(treatment_stats_customized['FailureToTreatPercentage'],
                                   treatment_stats_customized['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_customized['FailureToTreatPercentage'],
             treatment_stats_customized['UnnecessaryUsePercentage'], color='red',
             label=f'Customized All sites (AUC: {customized_auc:.4f})')
    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title('Failure to Treat vs. Unnecessary Treatment under different thresholds (2000-2022): Combined Random Results')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)

    return plt.gcf()

results_dict = {}
for num_clinics in [1, 5, 10]:
    random_sample_results = select_random_clinics_multiple_samples(df1, num_clinics, 1500)
    results_dict[num_clinics] = random_sample_results

plot = plot_combined_results(results_dict, treatment_stats_sum,treatment_stats_customized,prevalence_values)
plot.savefig('Figures/Random_Sampling_Combined_GISP.png')
plt.show()