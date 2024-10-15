import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from itertools import combinations
import random
import seaborn as sns

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
def calculate_treatment_stats_sum(data, prevalence_values,lambda_val):
    total = data['TOTAL'].sum()

    # Initialize arrays to store results
    failure_to_treat = np.zeros(len(prevalence_values))
    unnecessary_use = np.zeros(len(prevalence_values))

    for i, prevalence_value in enumerate(prevalence_values):
        drug_changed = False
        for _, row in data.iterrows():
            if not drug_changed and row['CipR_Sum_Prevalence'] >= prevalence_value:
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
        'FailureToTreatPercentage': failure_to_treat / total,
        'Expected Cost': unnecessary_use / total + lambda_val* failure_to_treat / total
        })


# customized sampling
def calculate_treatment_stats_sites(df, prevalence_values, lambda_val):
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
        clinic_data_stats = calculate_treatment_stats_sum(clinic_data, prevalence_values, lambda_val)
        clinic_data_stats['CLINIC'] = clinic
        clinic_total.append(clinic_data_stats)
    # append the result of each site together
    clinic_total = pd.concat(clinic_total, ignore_index=True)
    clinic_total.columns = ['Prevalence', 'FailureToTreat', 'UnnecessaryUse', 'Total',
                            'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Expected Cost', 'CLINIC']

    # sum the failure to treat and unnecessary under each prevalence
    clinic_sum = clinic_total.groupby('Prevalence')[['FailureToTreat', 'UnnecessaryUse']].sum().reset_index()
    clinic_sum['Total'] = df['TOTAL'].sum()
    # recalculate the failure to treat and unnecessary treatment percentage
    clinic_sum['UnnecessaryUsePercentage'] = clinic_sum['UnnecessaryUse'] / clinic_sum['Total']
    clinic_sum['FailureToTreatPercentage'] = clinic_sum['FailureToTreat'] / clinic_sum['Total']
    clinic_sum['Expected Cost'] = (clinic_sum['UnnecessaryUse'] + lambda_val * clinic_sum['FailureToTreat']) / \
                                  clinic_sum['Total']

    return clinic_sum

# calculate selected cost
def calculate_selected_cost(alpha, beta, num_samples=10000, threshold=None, lambda_val=1):
    if threshold is None:
        raise ValueError("Threshold must be provided")

    costs = np.zeros(len(alpha))
    non_zero_mask = alpha != 0
    non_zero_alpha = alpha[non_zero_mask]
    non_zero_beta = beta[non_zero_mask]

    if len(non_zero_alpha) > 0:
        samples = np.random.beta(non_zero_alpha, non_zero_beta, size=(num_samples, len(non_zero_alpha)))
        sample_costs = np.where(samples < threshold, lambda_val * non_zero_alpha, 1 - non_zero_alpha)
        non_zero_costs = np.mean(sample_costs, axis=0)
        costs[non_zero_mask] = non_zero_costs

    return costs

# calculate total cost
def calculate_total_cost(data, selected_indices):
    selected_cost = data.loc[selected_indices, 'selected_cost'].sum()
    not_selected_cost = data.loc[~data.index.isin(selected_indices), 'not_selected_cost'].sum()
    return selected_cost + not_selected_cost

# select the best site combination
def optimize_site_selection(data, num_sites):
    all_combinations = list(combinations(data.index, num_sites))
    min_cost = float('inf')
    best_combination = all_combinations[0]

    for combination in all_combinations:
        current_cost = calculate_total_cost(data, list(combination))
        if current_cost < min_cost:
            min_cost = current_cost
            best_combination = combination

    data['sites_chosen'] = 0
    data.loc[list(best_combination), 'sites_chosen'] = 1
    chosen_sites = data.loc[list(best_combination), 'CLINIC'].tolist()
    return {'data': data, 'min_cost': min_cost, 'chosen_sites': chosen_sites}

# update the data for chosen site
def update_data_for_chosen_sites(data_current, data_all_years):
    for _, row in data_current[data_current['sites_chosen'] == 1].iterrows():
        next_year = row['YEAR'] + 1
        clinic = row['CLINIC']

        next_year_data = data_all_years[(data_all_years['YEAR'] == next_year) & (data_all_years['CLINIC'] == clinic)]

        if not next_year_data.empty:
            data_current.loc[(data_current['CLINIC'] == clinic) & (data_current['YEAR'] == row['YEAR']), 'TOTAL'] = \
            next_year_data['TOTAL'].values[0]
            data_current.loc[(data_current['CLINIC'] == clinic) & (data_current['YEAR'] == row['YEAR']), 'CipR_Sum'] = \
            next_year_data['CipR_Sum'].values[0]
        else:
            print(f"Warning: No data found for clinic {clinic} in year {next_year}")

    return data_current


def adaptive_sampling(data, num_sites, threshold, num_samples=100, start_seed=10000, lambda_val=1):
    data = data.sort_values('YEAR')
    years = data['YEAR'].unique()
    all_samples_results = {}
    all_samples_selected_sites = pd.DataFrame()

    for sample in range(num_samples):
        seed = start_seed + sample
        results = {}
        selected_sites_df = pd.DataFrame()

        for year in years:
            random.seed(seed)
            print(f"Processing year: {year}, Sample: {sample + 1}")
            data_year = data[data['YEAR'] == year].copy()
            data_year['alpha'] = data_year['CipR_Sum'] / data_year['TOTAL']
            data_year['beta'] = 1 - data_year['alpha']
            data_year['not_selected_cost'] = data_year['alpha']
            data_year['selected_cost'] = calculate_selected_cost(
                data_year['alpha'].values,
                data_year['beta'].values,
                threshold=threshold,
                lambda_val=lambda_val
            )
            optimize_result = optimize_site_selection(data_year, num_sites)
            updated_data = update_data_for_chosen_sites(optimize_result['data'], data)
            results[str(year)] = {
                'data': updated_data,
                'min_cost': optimize_result['min_cost'],
                'chosen_sites': optimize_result['chosen_sites']
            }
            selected_sites_year = updated_data[updated_data['sites_chosen'] == 1][
                ['YEAR', 'CLINIC', 'TOTAL', 'CipR_Sum']]
            selected_sites_year['CipR_Sum_Prevalence'] = selected_sites_year['CipR_Sum'] / selected_sites_year['TOTAL']
            selected_sites_year['Threshold'] = threshold
            selected_sites_year['Sample'] = sample + 1
            selected_sites_year['Lambda'] = lambda_val
            selected_sites_df = pd.concat([selected_sites_df, selected_sites_year], ignore_index=True)

        all_samples_results[f'sample_{sample + 1}'] = results
        all_samples_selected_sites = pd.concat([all_samples_selected_sites, selected_sites_df], ignore_index=True)

    return {'results': all_samples_results, 'selected_sites': all_samples_selected_sites}


def adaptive_sampling_generation(data, num_sites_list, thresholds, lambda_list, prevalence_values):
    for lambda_val in lambda_list:
        all_lambda_stats = pd.DataFrame()

        for num_sites in num_sites_list:
            all_selected_sites = pd.DataFrame()
            print(f"Processing for {num_sites} sites, Lambda: {lambda_val}")

            for threshold in thresholds:
                sampling_result = adaptive_sampling(data, num_sites, threshold, lambda_val=lambda_val)
                selected_sites = sampling_result['selected_sites']
                selected_sites['NumSites'] = num_sites
                all_selected_sites = pd.concat([all_selected_sites, selected_sites], ignore_index=True)

            all_selected_sites.to_csv(
                f'Data/Min Cost {num_sites} sites selected with lambda {lambda_val}_GISP.csv')
            print(f"All samples generated for {num_sites} sites")


#calculate auc
def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


def calculate_treatment_stats_min_cost(all_sites, selected_sites_samples, prevalence_values, lambda_val):
    # Ensure the DataFrame is sorted by YEAR and CLINIC
    all_sites = all_sites.sort_values(['YEAR', 'CLINIC'])

    # Get unique years and clinics
    years = all_sites['YEAR'].unique()
    all_clinics = all_sites['CLINIC'].unique()

    # Initialize list to store all results
    all_results = []

    for sample_id in selected_sites_samples['Sample'].unique():
        sample_selected_sites = selected_sites_samples[selected_sites_samples['Sample'] == sample_id]

        for prevalence_value in prevalence_values:
            # Initialize dictionaries to store results for each clinic
            failure_to_treat = {clinic: 0 for clinic in all_clinics}
            unnecessary_use = {clinic: 0 for clinic in all_clinics}
            total = {clinic: 0 for clinic in all_clinics}

            for clinic in all_clinics:
                clinic_data = all_sites[all_sites['CLINIC'] == clinic]
                total[clinic] = clinic_data['TOTAL'].sum()

                drug_changed = False
                for year in years:
                    year_clinic = clinic_data[clinic_data['YEAR'] == year]
                    year_selected = sample_selected_sites[
                        (sample_selected_sites['YEAR'] == year) &
                        ((sample_selected_sites['Threshold'] - prevalence_value) < 1e-3)
                        ]

                    if clinic in year_selected['CLINIC'].values:
                        # For selected clinics, use their individual data
                        selected_clinic_data = year_selected[year_selected['CLINIC'] == clinic]
                        if not drug_changed and selected_clinic_data['CipR_Sum_Prevalence'].values[
                            0] >= prevalence_value:
                            drug_changed = True
                        if drug_changed:
                            unnecessary_use[clinic] += year_clinic['TOTAL'].sum() - year_clinic['CipR_Sum'].sum()
                        else:
                            failure_to_treat[clinic] += year_clinic['CipR_Sum'].sum()
                    else:
                        # For unselected clinics, base decision on estimated prevalence from selected sites
                        if not year_selected.empty and not year_clinic.empty:
                            estimated_prevalence = year_selected['CipR_Sum'].sum() / year_selected['TOTAL'].sum()
                            if not drug_changed and estimated_prevalence >= prevalence_value:
                                drug_changed = True
                            if drug_changed:
                                unnecessary_use[clinic] += year_clinic['TOTAL'].sum() - year_clinic['CipR_Sum'].sum()
                            else:
                                failure_to_treat[clinic] += year_clinic['CipR_Sum'].sum()

            # Calculate overall statistics
            total_sum = sum(total.values())
            failure_to_treat_sum = sum(failure_to_treat.values())
            unnecessary_use_sum = sum(unnecessary_use.values())

            # Get the number of sites for this sample
            num_sites = sample_selected_sites['NumSites'].iloc[0]

            all_results.append({
                'Sample': sample_id,
                'Prevalence': prevalence_value,
                'FailureToTreat': failure_to_treat_sum,
                'UnnecessaryUse': unnecessary_use_sum,
                'Total': total_sum,
                'UnnecessaryUsePercentage': unnecessary_use_sum / total_sum if total_sum > 0 else 0,
                'FailureToTreatPercentage': failure_to_treat_sum / total_sum if total_sum > 0 else 0,
                'Expected Cost': (unnecessary_use_sum + lambda_val * failure_to_treat_sum) / total_sum,
                'NumSites': num_sites
            })

    # Convert all results to DataFrame
    final_results = pd.DataFrame(all_results)

    return final_results
# plot unnecessary use and failure to treat
def plot_failure_vs_unnecessary(results, num_sites, lambda_val, treatment_stats_sum, treatment_stats_customized):
    plt.figure(figsize=(12, 6), dpi=700)

    sample_final = calculate_treatment_stats_min_cost(df, results, prevalence_values, lambda_val)
    samples = sample_final['Sample'].unique()

    for sample in samples:
        sample_data = sample_final[sample_final['Sample'] == sample]
        plt.plot(sample_data['FailureToTreatPercentage'], sample_data['UnnecessaryUsePercentage'],
                 color='gold', alpha=0.5)

    mean_results = sample_final.groupby('Prevalence')[['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
    mean_auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
    plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'],
             color='green', label=f'Mean of samples (AUC: {mean_auc:.4f})')

    # Add treatment_stats_sum plot
    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue', label=f'All sites uniform (AUC: {total_auc:.4f})')

    # Add treatment_stats_customized plot
    customized_auc = calculate_auc(treatment_stats_customized['FailureToTreatPercentage'],
                                   treatment_stats_customized['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_customized['FailureToTreatPercentage'],
             treatment_stats_customized['UnnecessaryUsePercentage'],
             color='red', label=f'All sites customized (AUC: {customized_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Failure to Treat vs. Unnecessary Treatment: {num_sites} sites, Lambda = {lambda_val}')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'FailureVsUnnecessary_{num_sites}sites_Lambda{lambda_val}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_failure_vs_unnecessary(all_results, treatment_stats_sum, treatment_stats_customized, lambda_val):
    plt.figure(figsize=(12, 6), dpi=700)

    color_palette = sns.color_palette("husl", n_colors=len(all_results['NumSites'].unique()))

    for i, num_sites in enumerate(sorted(all_results['NumSites'].unique())):
        data_subset = all_results[all_results['NumSites'] == num_sites]
        mean_results = data_subset.groupby('Prevalence')[
            ['FailureToTreatPercentage', 'UnnecessaryUsePercentage']].mean()
        auc = calculate_auc(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'])
        plt.plot(mean_results['FailureToTreatPercentage'], mean_results['UnnecessaryUsePercentage'],
                 color=color_palette[i], label=f'{num_sites} sites (AUC: {auc:.4f})')

    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue', label=f'All sites uniform (AUC: {total_auc:.4f})')

    customized_auc = calculate_auc(treatment_stats_customized['FailureToTreatPercentage'],
                                   treatment_stats_customized['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_customized['FailureToTreatPercentage'],
             treatment_stats_customized['UnnecessaryUsePercentage'],
             color='red', label=f'All sites customized (AUC: {customized_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Failure to Treat vs. Unnecessary Treatment: Combined Plot (Lambda = {lambda_val})')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'CombinedFailureVsUnnecessary_Lambda{lambda_val}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_vs_cost(all_results, treatment_stats_sum, treatment_stats_customized, lambda_val):
    plt.figure(figsize=(12, 8), dpi=700)

    color_palette = sns.color_palette("husl", n_colors=len(all_results['NumSites'].unique()))

    for i, num_sites in enumerate(sorted(all_results['NumSites'].unique())):
        data_subset = all_results[all_results['NumSites'] == num_sites]
        mean_results = data_subset.groupby('Prevalence')['Expected Cost'].mean()
        plt.plot(mean_results.index, mean_results.values, color=color_palette[i],
                 label=f'{num_sites} sites', marker='o')

    plt.plot(treatment_stats_sum['Prevalence'], treatment_stats_sum['Expected Cost'],
             color='blue', label='All sites uniform', marker='o')

    plt.plot(treatment_stats_customized['Prevalence'], treatment_stats_customized['Expected Cost'],
             color='red', label='All sites customized', marker='o')

    plt.xlabel('Prevalence')
    plt.ylabel('Expected Cost')
    plt.title(f'Expected Cost vs Prevalence (Lambda = {lambda_val})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'PrevalenceVsCost_Lambda{lambda_val}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
num_sites_list = [5, 10, 20]
lambda_val_list = [5]
prevalence_values = np.arange(0, 1.02, 0.02)
thresholds = np.arange(0, 1.02, 0.02)

adaptive_sampling_generation(df1, num_sites_list, thresholds, lambda_val_list, prevalence_values)

for lambda_val in lambda_val_list:
    all_lambda_stats = pd.DataFrame()

    # Calculate stats for all sites uniform and customized
    treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values, lambda_val)
    treatment_stats_customized = calculate_treatment_stats_sites(df, prevalence_values, lambda_val)

    for num_sites in num_sites_list:
        selected_sites = pd.read_csv(f'Data/Min Cost {num_sites} sites selected with lambda {lambda_val}_GISP.csv')
        treatment_stats = calculate_treatment_stats_min_cost(df, selected_sites, prevalence_values, lambda_val)
        all_lambda_stats = pd.concat([all_lambda_stats, treatment_stats], ignore_index=True)

        # Generate individual plots for each number of sites
        plot_failure_vs_unnecessary(selected_sites, num_sites, lambda_val, treatment_stats_sum,
                                    treatment_stats_customized)

    # Generate combined plots
    plot_combined_failure_vs_unnecessary(all_lambda_stats, treatment_stats_sum, treatment_stats_customized, lambda_val)
    plot_prevalence_vs_cost(all_lambda_stats, treatment_stats_sum, treatment_stats_customized, lambda_val)

    print(f"Plots generated for Lambda: {lambda_val}")

