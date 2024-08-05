import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random


# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(df.columns[[0]], axis=1)
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(dt.columns[[0]], axis=1)


def calculate_treatment_stats_sum(data, prevalence_values):
    results = []
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

        results.append({
            'Prevalence': prevalence_values,
            'FailureToTreat': failure_to_treat,
            'UnnecessaryUse': unnecessary_use,
            'Total': total,
            'UnnecessaryUsePercentage': unnecessary_use / total,
            'FailureToTreatPercentage': failure_to_treat / total
        })
    return pd.DataFrame(results)


def preprocess_data(data):
    min_year = data['YEAR'].min()
    max_year = data['YEAR'].max()
    total_years = max_year - min_year + 1

    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index

    return data[data['CLINIC'].isin(sites_with_all_years)]


df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipRsum_prevalence'] = dt1['CipRsum'] / dt1['TOTAL']


def calculate_total_cost(data, selected_indices):
    selected_cost = data.loc[selected_indices, 'selected_cost'].sum()
    not_selected_cost = data.loc[~data.index.isin(selected_indices), 'not_selected_cost'].sum()
    return selected_cost + not_selected_cost


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


def update_data_for_chosen_sites(data_current, data_all_years):
    for _, row in data_current[data_current['sites_chosen'] == 1].iterrows():
        next_year = row['YEAR'] + 1
        clinic = row['CLINIC']

        next_year_data = data_all_years[(data_all_years['YEAR'] == next_year) & (data_all_years['CLINIC'] == clinic)]

        if not next_year_data.empty:
            data_current.loc[(data_current['CLINIC'] == clinic) & (data_current['YEAR'] == row['YEAR']), 'TOTAL'] = \
            next_year_data['TOTAL'].values[0]
            data_current.loc[(data_current['CLINIC'] == clinic) & (data_current['YEAR'] == row['YEAR']), 'CipRsum'] = \
            next_year_data['CipRsum'].values[0]
        else:
            print(f"Warning: No data found for clinic {clinic} in year {next_year}")

    return data_current


def calculate_lambda(data, threshold):
    data_sorted = data.sort_values('FailureToTreatPercentage')
    idx = np.searchsorted(data_sorted['FailureToTreatPercentage'], threshold)

    if idx == 0:
        idx = 1
    elif idx == len(data_sorted):
        idx = len(data_sorted) - 1

    point1 = data_sorted.iloc[idx - 1]
    point2 = data_sorted.iloc[idx]

    slope = (point2['UnnecessaryUsePercentage'] - point1['UnnecessaryUsePercentage']) / \
            (point2['FailureToTreatPercentage'] - point1['FailureToTreatPercentage'])

    lambda_value = -1 * slope

    return lambda_value

def calculate_lambda_fitting(threshold):
    if threshold<=0.16:
        slope=-262.36534061*threshold**2*3+ 92.36681857*threshold*2-12.51689277
    else:
        slope= 0
    lambda_value = -1 * slope
    return lambda_value

def calculate_selected_cost(alpha, beta, num_samples=10000, threshold=None):
    if threshold is None:
        raise ValueError("Threshold must be provided")

    costs = np.zeros(len(alpha))
    lambda_val = calculate_lambda_fitting(threshold=threshold)

    non_zero_mask = alpha != 0
    non_zero_alpha = alpha[non_zero_mask]
    non_zero_beta = beta[non_zero_mask]

    if len(non_zero_alpha) > 0:
        samples = np.random.beta(non_zero_alpha, non_zero_beta, size=(num_samples, len(non_zero_alpha)))
        sample_costs = np.where(samples < threshold, non_zero_alpha, lambda_val * (1 - non_zero_alpha))
        non_zero_costs = np.mean(sample_costs, axis=0)
        costs[non_zero_mask] = non_zero_costs

    return costs


def adaptive_sampling(data, num_sites, threshold, seed=573):
    data = data.sort_values('YEAR')
    years = data['YEAR'].unique()

    results = {}
    selected_sites_df = pd.DataFrame()

    for year in years:
        random.seed(seed)
        print(f"Processing year: {year}")

        data_year = data[data['YEAR'] == year].copy()
        data_year['alpha'] = data_year['CipRsum'] / data_year['TOTAL']
        data_year['beta'] = 1 - data_year['alpha']
        data_year['not_selected_cost'] = data_year['alpha']
        data_year['selected_cost'] = calculate_selected_cost(
            data_year['alpha'].values,
            data_year['beta'].values,
            threshold=threshold
        )

        optimize_result = optimize_site_selection(data_year, num_sites)
        updated_data = update_data_for_chosen_sites(optimize_result['data'], data)

        results[str(year)] = {
            'data': updated_data,
            'min_cost': optimize_result['min_cost'],
            'chosen_sites': optimize_result['chosen_sites']
        }

        selected_sites_year = updated_data[updated_data['sites_chosen'] == 1][['YEAR', 'CLINIC', 'TOTAL', 'CipRsum']]
        selected_sites_year['CipRsum_prevalence'] = selected_sites_year['CipRsum'] / selected_sites_year['TOTAL']
        selected_sites_year['Threshold'] = threshold
        selected_sites_df = pd.concat([selected_sites_df, selected_sites_year], ignore_index=True)

    return {'results': results, 'selected_sites': selected_sites_df}


def calculate_point_for_threshold(data, num_sites, threshold):
    result = adaptive_sampling(data=data, num_sites=num_sites, threshold=threshold)['selected_sites']
    dt_new = dt.copy()
    data_agg = result.groupby('YEAR').sum().reset_index()
    data_agg['CipRsum_prevalence'] = data_agg['CipRsum'] / data_agg['TOTAL']
    dt_new['CipRsum_prevalence'] = data_agg['CipRsum_prevalence'].values

    total = dt_new['TOTAL'].sum()

    drug_change = (dt_new['CipRsum_prevalence'] >= threshold).astype(int)
    failure_to_treat = ((1 - drug_change) * dt_new['CipRsum']).sum()
    unnecessary_use = (drug_change * (dt_new['TOTAL'] - dt_new['CipRsum'])).sum()

    return failure_to_treat / total, unnecessary_use / total, total, result


def adaptive_sampling_and_plot_varying_threshold(df1, prevalence_values, num_sites_list, threshold_values):
    treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
    treatment_stats_sum['Method'] = 'Total'
    plot_data = {'Total': treatment_stats_sum[
        ['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage']].values.tolist()}

    results = []
    all_selected_sites = pd.DataFrame()
    for num_sites in num_sites_list:
        plot_data[f"Top {num_sites} sites"] = []

        for threshold in threshold_values:
            failure_to_treat, unnecessary_use, total, selected_sites = calculate_point_for_threshold(df1, num_sites, threshold)
            plot_data[f"Top {num_sites} sites"].append([threshold, unnecessary_use, failure_to_treat])
            results.append({
                'Site number': num_sites,
                'Prevalence': threshold,
                'Total': total,
                'UnnecessaryUsePercentage': unnecessary_use,
                'FailureToTreatPercentage': failure_to_treat
            })
            selected_sites['NumSites'] = num_sites
            all_selected_sites = pd.concat([all_selected_sites, selected_sites], ignore_index=True)
    plt.figure(figsize=(12, 6))
    results = pd.DataFrame(results)
    results.to_csv("Data/PrevalenceWithUandFinPopulation.csv", index=False)
    all_selected_sites.to_csv("Data/All_Selected_Sites_Population.csv", index=False)
    auc_results = []
    for method, data in plot_data.items():
        data_sorted = sorted(data, key=lambda x: x[2])  # Sort by FailureToTreatPercentage
        x = [point[2] for point in data_sorted]
        y = [point[1] for point in data_sorted]

        auc = np.trapz(y, x)
        auc_normalized = auc / (max(x) * max(y))
        auc_results.append({'Method': method, 'AUC': auc_normalized})

        plt.plot(x, y, label=f"{method} (AUC: {auc_normalized:.4f})")

    plt.title("Min Cost Strategy (Varying Threshold)")
    plt.xlabel("Failure to Treat (%)")
    plt.ylabel("Unnecessary Treatment (%)")
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    plt.savefig('Figures/Min Cost Strategy for total population varying threshold.png')
    plt.show()

    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv('Data/auc_results_sample_varying_threshold.csv', index=False)

    return auc_df

# Set the parameters and run the analysis
prevalence_values = np.arange(0, 1.02, 0.02)
num_sites_list = [5, 10]

# Calculate treatment_stats_sum before using it in other functions
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
treatment_stats_sum['Sample'] = 'Total'
threshold_values = np.arange(0, 1.02, 0.02) #to be adjusted
treatment_stats_sum.to_csv('Data/sample_curve_results.csv',index=False)

auc_results = adaptive_sampling_and_plot_varying_threshold(df1, prevalence_values, num_sites_list,threshold_values)
print(auc_results)

