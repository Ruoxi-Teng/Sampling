import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import numpy as np
import random

df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(df.columns[[0]], axis=1)
df['CipRsum_prevalence'] = df['CipRsum']/df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(dt.columns[[0]], axis=1)

# suppose we only include the sites that are sampled every year

def preprocess_data(data):


# Calculate min and max years
    min_year = data['YEAR'].min()
    max_year = data['YEAR'].max()
    total_years = max_year - min_year + 1

    # Filter out sites that are not sampled every year
    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index

    filtered_data = data[data['CLINIC'].isin(sites_with_all_years)]

    return filtered_data


df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipRsum_prevalence'] = dt1['CipRsum']/dt1['TOTAL']
# print(df1)
# print(dt1)


def calculate_total_cost(data, selected_indices):

    selected_cost = data.loc[selected_indices, 'selected_cost'].sum()
    not_selected_cost = data.loc[~data.index.isin(selected_indices), 'not_selected_cost'].sum()
    return selected_cost + not_selected_cost
def optimize_site_selection(data, num_sites):

    all_combinations = list(combinations(data.index, num_sites))

    min_cost = float('inf')
    best_combination = None

    for combination in all_combinations:

        current_cost = calculate_total_cost(data,list(combination))

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


def calculate_selected_cost(alpha, beta, lambda_val=0.5, num_samples=10000, threshold=0.05):
    # Initialize an array to store costs
    costs = np.zeros(len(alpha))

    # Handle non-zero alpha values
    non_zero_mask = alpha != 0
    non_zero_alpha = alpha[non_zero_mask]
    non_zero_beta = beta[non_zero_mask]
    if len(non_zero_alpha) > 0:
        # Generate samples from the beta distribution for non-zero alpha values
        samples = np.random.beta(non_zero_alpha, non_zero_beta, size=(num_samples, len(non_zero_alpha)))

        # Calculate cost for each sample
        sample_costs = np.where(samples < threshold, non_zero_alpha, lambda_val * (1 - non_zero_alpha))

        # Calculate mean cost for each non-zero alpha
        non_zero_costs = np.mean(sample_costs, axis=0)

        # Assign calculated costs back to the original array
        costs[non_zero_mask] = non_zero_costs

    return costs

def adaptive_sampling(data, num_sites, seed=573):
    # Sort data by year

    data = data.sort_values('YEAR')
    years = data['YEAR'].unique()

    results = {}
    selected_sites_df = pd.DataFrame()

    for year in years:
        random.seed(seed)
        print(f"Processing year: {year}")

        # Filter data for the current year
        data_year = data[data['YEAR'] == year].copy()

        # Calculate alpha and beta with a small epsilon to avoid zeros
        data_year['alpha'] = data_year['CipRsum'] / data_year['TOTAL']
        data_year['beta'] = 1 - data_year['alpha']

        # Calculate not_selected_cost, it is the mean of the beta distribution
        data_year['not_selected_cost'] = data_year['alpha']

        # Calculate sampled_cost
        data_year['selected_cost'] = calculate_selected_cost(
            data_year['alpha'].values,
            data_year['beta'].values
        )
        # Optimize site selection
        optimize_result = optimize_site_selection(data_year, num_sites)

        # Update data for chosen sites
        updated_data = update_data_for_chosen_sites(optimize_result['data'], data)

        # Store results
        results[str(year)] = {
            'data': updated_data,
            'min_cost': optimize_result['min_cost'],
            'chosen_sites': optimize_result['chosen_sites']
        }

        # Add selected sites to the new dataframe
        selected_sites_year = updated_data[updated_data['sites_chosen'] == 1][['YEAR', 'CLINIC', 'TOTAL', 'CipRsum']]
        selected_sites_year['CipRsum_prevalence'] = selected_sites_year['CipRsum'] / selected_sites_year['TOTAL']
        selected_sites_df = pd.concat([selected_sites_df, selected_sites_year], ignore_index=True)

        # Update the main dataset with the new values for chosen sites
        data = data.merge(
            updated_data[updated_data['sites_chosen'] == 1][['CLINIC', 'YEAR', 'TOTAL', 'CipRsum']],
            on=['CLINIC', 'YEAR'], how='left', suffixes=('', '_new')
        )
        data['TOTAL'] = data['TOTAL_new'].fillna(data['TOTAL'])
        data['CipRsum'] = data['CipRsum_new'].fillna(data['CipRsum'])
        data = data.drop(columns=['TOTAL_new', 'CipRsum_new'])

    return {'results': results, 'selected_sites': selected_sites_df}

# Run adaptive_sampling for different numbers of sites
result5 = adaptive_sampling(data=df1, num_sites=5)['selected_sites']
result10 = adaptive_sampling(data=df1, num_sites=10)['selected_sites']

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
result1 = calculate_treatment_stats_sum(result5, prevalence_values)
result2 = calculate_treatment_stats_sum(result10, prevalence_values)
# Add 'Sample' column to results
result1['Sample'] = "Top 5 sites"
result2['Sample'] = "Top 10 sites"

# Combine results
plot_cost = pd.concat([
    treatment_stats_sum[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Sample']],
    result1[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Sample']],
    result2[['Prevalence', 'UnnecessaryUsePercentage', 'FailureToTreatPercentage', 'Sample']]
], ignore_index=True)

# Rename 'Sample' column to 'Method'
plot_cost = plot_cost.rename(columns={'Sample': 'Method'})

# Create the plot
plt.figure(figsize=(12, 6))

for method in plot_cost['Method'].unique():
    data = plot_cost[plot_cost['Method'] == method]
    plt.plot(data['FailureToTreatPercentage'], data['UnnecessaryUsePercentage'], label=method)

plt.title("Min Cost Strategy")
plt.xlabel("Failure to Treat (%)")
plt.ylabel("Unnecessary Treatment (%)")
plt.legend()
plt.grid(True)

# Format axes to display percentages
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

plt.savefig('Figures/Min Cost Strategy lambda = 0.5 threshold = 0.05.png')
plt.show()
