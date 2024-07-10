import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random
from scipy.interpolate import interp1d
from scipy.integrate import trapz

# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(df.columns[[0]], axis=1)
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(dt.columns[[0]], axis=1)


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

prevalence_values = np.arange(0, 1.002, 0.002)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
treatment_stats_sum['Sample'] = 'Total'
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
    best_combination = None

    for combination in all_combinations:
        current_cost = calculate_total_cost(data, list(combination))
        if current_cost < min_cost:
            min_cost = current_cost
            best_combination = combination

    data['sites_chosen'] = 0
    data.loc[list(best_combination), 'sites_chosen'] = 1
    chosen_sites = data.loc[list(best_combination), 'CLINIC'].tolist()

    return {'data': data, 'min_cost': min_cost, 'chosen_sites': chosen_sites}


def update_data_for_chosen_sites(data_2000, data_all_years):
    for _, row in data_2000[data_2000['sites_chosen'] == 1].iterrows():
        next_year = row['YEAR'] + 1
        clinic = row['CLINIC']

        next_year_data = data_all_years[(data_all_years['YEAR'] == next_year) & (data_all_years['CLINIC'] == clinic)]

        if not next_year_data.empty:
            data_2000.loc[(data_2000['CLINIC'] == clinic) & (data_2000['YEAR'] == row['YEAR']), 'TOTAL'] = \
            next_year_data['TOTAL'].values[0]
            data_2000.loc[(data_2000['CLINIC'] == clinic) & (data_2000['YEAR'] == row['YEAR']), 'CipRsum'] = \
            next_year_data['CipRsum'].values[0]
        else:
            print(f"Warning: No data found for clinic {clinic} in year {next_year}")

    return data_2000


def calculate_lambda(data, threshold):
    interp_func = interp1d(data['FailureToTreatPercentage'],
                           data['UnnecessaryUsePercentage'],
                           kind='linear')

    epsilon = 1e-6
    y1 = interp_func(threshold - epsilon)
    y2 = interp_func(threshold + epsilon)
    slope = (y2 - y1) / (2 * epsilon)

    return -1 / slope if slope != 0 else np.inf


def calculate_selected_cost(alpha, beta, num_samples=10000, threshold=0.05):
    costs = np.zeros(len(alpha))
    lambda_val = calculate_lambda(data=treatment_stats_sum, threshold=threshold)

    non_zero_mask = alpha != 0
    non_zero_alpha = alpha[non_zero_mask]
    non_zero_beta = beta[non_zero_mask]

    if len(non_zero_alpha) > 0:
        samples = np.random.beta(non_zero_alpha, non_zero_beta, size=(num_samples, len(non_zero_alpha)))
        sample_costs = np.where(samples < threshold, non_zero_alpha, lambda_val * (1 - non_zero_alpha))
        non_zero_costs = np.mean(sample_costs, axis=0)
        costs[non_zero_mask] = non_zero_costs

    return costs


def adaptive_sampling(data, num_sites, seed=573):
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
            data_year['beta'].values
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
        selected_sites_df = pd.concat([selected_sites_df, selected_sites_year], ignore_index=True)

        data = data.merge(
            updated_data[updated_data['sites_chosen'] == 1][['CLINIC', 'YEAR', 'TOTAL', 'CipRsum']],
            on=['CLINIC', 'YEAR'], how='left', suffixes=('', '_new')
        )
        data['TOTAL'] = data['TOTAL_new'].fillna(data['TOTAL'])
        data['CipRsum'] = data['CipRsum_new'].fillna(data['CipRsum'])
        data = data.drop(columns=['TOTAL_new', 'CipRsum_new'])

    return {'results': results, 'selected_sites': selected_sites_df}


def adaptive_sampling_and_plot(df1, prevalence_values, threshold, num_sites_list):
    treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
    treatment_stats_sum['Method'] = 'Total'

    plot_cost = [treatment_stats_sum[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Method']]]

    for num_sites in num_sites_list:
        result = adaptive_sampling(data=df1, num_sites=num_sites)['selected_sites']
        result_stats = calculate_treatment_stats_sum(result, prevalence_values)
        result_stats['Method'] = f"Top {num_sites} sites"
        plot_cost.append(result_stats[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Method']])

    plot_cost = pd.concat(plot_cost, ignore_index=True)

    def calculate_auc(x, y):
        # Ensure the data is sorted by x values
        sorted_data = sorted(zip(x, y))
        x_sorted, y_sorted = zip(*sorted_data)

        # Calculate AUC using trapezoidal rule
        auc = trapz(y_sorted, x_sorted)

        # Normalize AUC to be between 0 and 1
        auc_normalized = auc / (max(x_sorted) * max(y_sorted))

        return auc_normalized

    plt.figure(figsize=(12, 6))

    auc_results = []
    for method in plot_cost['Method'].unique():
        data = plot_cost[plot_cost['Method'] == method].sort_values('FailureToTreatPercentage')
        auc = calculate_auc(data['FailureToTreatPercentage'], data['UnnecessaryUsePercentage'])
        auc_results.append({'Method': method, 'AUC': auc})
        plt.plot(data['FailureToTreatPercentage'], data['UnnecessaryUsePercentage'],
                 label=f"{method} (AUC: {auc:.4f})")

    plt.title(f"Min Cost Strategy (Threshold: {threshold:.2f})")
    plt.xlabel("Failure to Treat (%)")
    plt.ylabel("Unnecessary Treatment (%)")
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    plt.savefig(f'Figures/Min Cost Strategy for sampled population threshold = {threshold:.2f}.png')
    plt.show()

    auc_df = pd.DataFrame(auc_results)
    auc_df['Threshold'] = threshold
    auc_df.to_csv(f'Data/auc_results_sample_threshold_{threshold:.2f}.csv', index=False)

    return auc_df


# Set the threshold and run the analysis
threshold = 0.01
num_sites_list = [5, 10]
prevalence_values = np.arange(0, 1.002, 0.002)

auc_results = adaptive_sampling_and_plot(df1, prevalence_values, threshold, num_sites_list)

print(auc_results)