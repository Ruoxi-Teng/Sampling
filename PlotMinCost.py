import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from itertools import combinations
import numpy as np
dt=pd.read_csv("Data/CIP_summary.csv")


def adaptive_sampling(data, num_sites=10, lambda_val=1, seed=573):
    # Sort data by year
    data = data.sort_values('YEAR')
    years = data['YEAR'].unique()

    results = {}
    selected_sites_df = pd.DataFrame()

    for year in years:
        print(f"Processing year: {year}")

        # Filter data for the current year
        data_year = data[data['YEAR'] == year].copy()

        # Calculate alpha and beta with a small epsilon to avoid zeros
        epsilon = 1e-10
        data_year['alpha'] = data_year['CipRsum'] / data_year['TOTAL'] + epsilon
        data_year['beta'] = 1 - data_year['alpha']

        # Calculate not_selected_cost
        np.random.seed(seed + year)
        n_samples = 10000
        alpha_values = np.maximum(data_year['alpha'].values, epsilon)
        beta_values = np.maximum(data_year['beta'].values, epsilon)
        random_samples = beta.rvs(alpha_values[:, np.newaxis], beta_values[:, np.newaxis],
                                  size=(len(data_year), n_samples))
        data_year['not_selected_cost'] = np.mean(random_samples, axis=1)

        # Calculate sampled_cost
        data_year['p'] = beta.cdf(0.05, data_year['alpha'], data_year['beta'])
        data_year['sampled_cost'] = data_year['p'] * data_year['alpha'] + \
                                    (1 - data_year['p']) * (1 - data_year['alpha']) * lambda_val

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
        selected_sites_year = updated_data[updated_data['sites_chosen'] == 1][['YEAR', 'clinic', 'TOTAL', 'CipRsum']]
        selected_sites_year['prevalence'] = selected_sites_year['CipRsum'] / selected_sites_year['TOTAL']
        selected_sites_df = pd.concat([selected_sites_df, selected_sites_year], ignore_index=True)

        # Update the main dataset with the new values for chosen sites
        data = data.merge(
            updated_data[updated_data['sites_chosen'] == 1][['clinic', 'YEAR', 'TOTAL', 'CipRsum']],
            on=['clinic', 'YEAR'], how='left', suffixes=('', '_new')
        )
        data['TOTAL'] = data['TOTAL_new'].fillna(data['TOTAL'])
        data['CipRsum'] = data['CipRsum_new'].fillna(data['CipRsum'])
        data = data.drop(columns=['TOTAL_new', 'CipRsum_new'])

    return {'results': results, 'selected_sites': selected_sites_df}


def update_data_for_chosen_sites(data_2000, data_all_years):
    for _, row in data_2000[data_2000['sites_chosen'] == 1].iterrows():
        next_year = row['YEAR'] + 1
        clinic = row['clinic']

        next_year_data = data_all_years[(data_all_years['YEAR'] == next_year) & (data_all_years['clinic'] == clinic)]

        if not next_year_data.empty:
            data_2000.loc[(data_2000['clinic'] == clinic) & (data_2000['YEAR'] == row['YEAR']), 'TOTAL'] = \
            next_year_data['TOTAL'].values[0]
            data_2000.loc[(data_2000['clinic'] == clinic) & (data_2000['YEAR'] == row['YEAR']), 'CipRsum'] = \
            next_year_data['CipRsum'].values[0]
        else:
            print(f"Warning: No data found for clinic {clinic} in year {next_year}")

    return data_2000


def optimize_site_selection(data, num_sites):
    def calculate_total_cost(selected_indices):
        selected_cost = data.loc[selected_indices, 'sampled_cost'].sum()
        not_selected_cost = data.loc[~data.index.isin(selected_indices), 'not_selected_cost'].sum()
        return selected_cost + not_selected_cost

    all_combinations = list(combinations(data.index, num_sites))

    min_cost = float('inf')
    best_combination = None

    for combination in all_combinations:
        current_cost = calculate_total_cost(combination)
        if current_cost < min_cost:
            min_cost = current_cost
            best_combination = combination

    data['sites_chosen'] = 0
    data.loc[list(best_combination), 'sites_chosen'] = 1
    chosen_sites = data.loc[list(best_combination), 'clinic'].tolist()

    return {'data': data, 'min_cost': min_cost, 'chosen_sites': chosen_sites}

# we only use the data that occurs each year
def preprocess_data(data):
    # Delete the first two rows
    data = data.drop(data.columns[[0]],axis=1)

    # Calculate min and max years
    min_year = data['YEAR'].min()
    max_year = data['YEAR'].max()
    total_years = max_year - min_year + 1

    # Filter out sites that are not sampled every year
    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index

    filtered_data = data[data['CLINIC'].isin(sites_with_all_years)]

    return filtered_data


processed_data = preprocess_data(dt)
print(processed_data)


result=adaptive_sampling(data=processed_data,num_sites=10)

print(result['data'])