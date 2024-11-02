import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

# Load and preprocess data
# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(columns=df.columns[0])
df['CipR_Sum_prevalence'] = df['CipR_Sum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(columns=dt.columns[0])

# Load and preprocess data
# df = pd.read_csv("Data/site_prevalence_data.csv")
# dt = pd.read_csv("Data/total_prevalence_data.csv")

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


# customized sampling
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


# first select the samples
def select_random_clinics_multiple_samples(df, num_clinics, num_samples):
    # Ensure the DataFrame is sorted by YEAR
    df = df.sort_values('YEAR')

    # Get unique years
    years = df['YEAR'].unique()

    # Initialize an empty list to store results from all samples
    all_samples_results = []

    for sample in range(1, num_samples + 1):
        np.random.seed(10000 + sample)
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

        result_df['Sample'] = sample

        # Append to the list of all samples results
        all_samples_results.append(result_df)

    # Concatenate results from all samples
    final_result = pd.concat(all_samples_results, ignore_index=True)

    return final_result

# newest version fixed sampling calculation
# adjust for random sampling
def calculate_treatment_stats_threshold_samples(all_sites, selected_sites_samples, prevalence_values):
    # Ensure the DataFrame is sorted by YEAR and CLINIC
    all_sites = all_sites.sort_values(['YEAR', 'CLINIC'])

    # Get unique years and clinics
    years = all_sites['YEAR'].unique()
    all_clinics = all_sites['CLINIC'].unique()

    # Initialize DataFrame to store final results
    final_results = pd.DataFrame()

    for sample_id in selected_sites_samples['Sample'].unique():
        sample_selected_sites = selected_sites_samples[selected_sites_samples['Sample'] == sample_id]

        results_list = []

        for prevalence_value in prevalence_values:
            for clinic in all_clinics:
                clinic_data = all_sites[all_sites['CLINIC'] == clinic]
                total = clinic_data['TOTAL'].sum()
                failure_to_treat = 0
                unnecessary_use = 0

                for year in years:
                    year_clinic = clinic_data[clinic_data['YEAR'] == year]
                    if year_clinic.empty:
                        continue

                    year_selected_sites = sample_selected_sites[sample_selected_sites['YEAR'] == year]

                    if clinic in year_selected_sites['CLINIC'].values:
                        # For selected clinics, use their individual data
                        selected_clinic_data = year_selected_sites[year_selected_sites['CLINIC'] == clinic]
                        if not selected_clinic_data.empty:
                            if selected_clinic_data['CipR_Sum_prevalence'].values[0] >= prevalence_value:
                                unnecessary_use += year_clinic['TOTAL'].sum() - year_clinic['CipR_Sum'].sum()
                            else:
                                failure_to_treat += year_clinic['CipR_Sum'].sum()
                    else:
                        # For unselected clinics, base decision on estimated prevalence from selected sites
                        if not year_selected_sites.empty:
                            estimated_prevalence = year_selected_sites['CipR_Sum'].sum() / year_selected_sites[
                                'TOTAL'].sum()
                            if estimated_prevalence >= prevalence_value:
                                unnecessary_use += year_clinic['TOTAL'].sum() - year_clinic['CipR_Sum'].sum()
                            else:
                                failure_to_treat += year_clinic['CipR_Sum'].sum()

                results_list.append({
                    'Sample': sample_id,
                    'CLINIC': clinic,
                    'Prevalence': prevalence_value,
                    'FailureToTreat': failure_to_treat,
                    'UnnecessaryUse': unnecessary_use,
                    'Total': total
                })

        # Convert results list to DataFrame
        sample_results = pd.DataFrame(results_list)

        # Sum the failure to treat and unnecessary use under each prevalence for this sample
        sample_sum = sample_results.groupby(['Sample', 'Prevalence'])[
            ['FailureToTreat', 'UnnecessaryUse', 'Total']].sum().reset_index()

        # Recalculate the failure to treat and unnecessary treatment percentage
        sample_sum['UnnecessaryUsePercentage'] = sample_sum['UnnecessaryUse'] / sample_sum['Total']
        sample_sum['FailureToTreatPercentage'] = sample_sum['FailureToTreat'] / sample_sum['Total']

        # Append to final results
        final_results = pd.concat([final_results, sample_sum], ignore_index=True)

    return final_results

#calculate auc
def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


# plot random sampling result
def plot_random_results(results, treatment_stats_sum, treatment_stats_customized, num_clinics):
    plt.figure(figsize=(9, 4), dpi=700)

    # plot each sample curve
    sample_final = calculate_treatment_stats_threshold_samples(df, results, prevalence_values)
    samples = sample_final['Sample'].unique()

    # Plot each sample as a line
    for sample in samples:
        sample_data = sample_final[sample_final['Sample'] == sample]
        plt.plot(sample_data['FailureToTreatPercentage'], sample_data['UnnecessaryUsePercentage'], color='gold',
                 alpha=0.5)  # alpha for transparency

    # plot mean result
    mean_results = sample_final.groupby('Prevalence')[['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
    plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'], color='green',
             label=f'Mean of samples (AUC: {mean_auc:.4f})')

    # plot all sites curve
    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    # plot customized switch curve
    customized_auc = calculate_auc(treatment_stats_customized['FailureToTreatPercentage'],
                                   treatment_stats_customized['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_customized['FailureToTreatPercentage'],
             treatment_stats_customized['UnnecessaryUsePercentage'], color='red',
             label=f'Customized All sites (AUC: {customized_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(
        f'Failure to Treat v.s. Unnecessary Treatment (2000-2022): Random {num_clinics} sites samples Customized')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf()

prevalence_values=np.arange(0,1.02,0.02)
treatment_stats_sum=calculate_treatment_stats_sum(dt,prevalence_values)
treatment_stats_customized=calculate_treatment_stats_sites(df,prevalence_values)

# result: fixed sampling
for num_clinics in [1, 5, 10]:
    fixed_sample_results = select_random_clinics_multiple_samples(df1, num_clinics, 100)
    plot = plot_random_results(fixed_sample_results,  treatment_stats_sum, treatment_stats_customized,num_clinics)
    plot.savefig(f'Figures/Random {num_clinics} sites Customized_GISP.png')
    plt.show()


def plot_combined_random_results_simplified(results_dict, treatment_stats_sum, treatment_stats_customized):
    plt.figure(figsize=(9, 4), dpi=700)

    colors = ['orange', 'yellow', 'green']

    # Plot mean results for each number of clinics
    for i, (num_clinics, results) in enumerate(results_dict.items()):
        sample_final = calculate_treatment_stats_threshold_samples(df, results, prevalence_values)

        # Calculate and plot mean result
        mean_results = sample_final.groupby('Prevalence')[
            ['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
        mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
        plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'],
                 color=colors[i], label=f'Mean of {num_clinics} clinics (AUC: {mean_auc:.4f})')

    # Plot all sites curve (treatment_stats_sum)
    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue', label=f'All sites (AUC: {total_auc:.4f})')

    # Plot customized switch curve (treatment_stats_customized)
    customized_auc = calculate_auc(treatment_stats_customized['FailureToTreatPercentage'],
                                   treatment_stats_customized['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_customized['FailureToTreatPercentage'],
             treatment_stats_customized['UnnecessaryUsePercentage'], color='red',
             label=f'Customized All sites (AUC: {customized_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title('Failure to Treat vs. Unnecessary Treatment (2000-2022): Combined Random Customized Results')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf()


# Usage
prevalence_values = np.arange(0, 1.02, 0.02)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
treatment_stats_customized = calculate_treatment_stats_sites(df, prevalence_values)

results_dict = {}
for num_clinics in [1, 5, 10]:
    random_sample_results = select_random_clinics_multiple_samples(df1, num_clinics, 100)
    results_dict[num_clinics] = random_sample_results

plot = plot_combined_random_results_simplified(results_dict, treatment_stats_sum, treatment_stats_customized)
plot.savefig('Figures/Combined_Random_Samples_Customized_GISP.png')
plt.show()