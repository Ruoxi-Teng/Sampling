import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from itertools import combinations
import random

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

prevalence_values=np.arange(0,1.02,0.02)
treatment_stats_sum=calculate_treatment_stats_sum(dt,prevalence_values)

# test model fitting
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def inverse(x, k):
    return k / x

def exponential(x, a, b):
    return a * np.exp(-b * x)

def calculate_aic(n, mse, num_params):
    return 2 * num_params + n * np.log(mse)

# fitting results
# Prepare data
x = treatment_stats_sum['FailureToTreatPercentage'].values
y = treatment_stats_sum['UnnecessaryUsePercentage'].values

# Fit models
models = {
    'Quadratic': (quadratic, 3),
    'Cubic': (cubic, 4),
    'Inverse': (inverse, 1),
    'Exponential': (exponential, 2)
}

results = {}
for name, (func, num_params) in models.items():
    try:
        popt, _ = curve_fit(func, x, y)
        y_pred = func(x, *popt)
        mse = mean_squared_error(y, y_pred)
        aic = calculate_aic(len(x), mse, num_params)
        results[name] = {'mse': mse, 'aic': aic, 'params': popt, 'func': func}
    except:
        print(f"Could not fit {name} model")

# Print results
for name, res in results.items():
    print(f"{name}:")
    print(f"  MSE: {res['mse']:.6f}")
    print(f"  AIC: {res['aic']:.6f}")
    print(f"  Parameters: {res['params']}")
    print()

# Find best model
best_model = min(results, key=lambda x: results[x]['aic'])
print(f"Best model according to AIC: {best_model}")

# Calculate fitted values, slope, and lambda for the best model
best_func = results[best_model]['func']
best_params = results[best_model]['params']

fitted_values = best_func(x, *best_params)

# Calculate fitted values, slope, and lambda for the best model
best_func = results[best_model]['func']
best_params = results[best_model]['params']

fitted_values = best_func(x, *best_params)

# Calculate slope
if best_model == 'Quadratic':
    a, b, _ = best_params
    slope = 2 * a * x + b
elif best_model == 'Cubic':
    a, b, c, _ = best_params
    slope = 3 * a * x**2 + 2 * b * x + c
elif best_model == 'Inverse':
    k, = best_params
    slope = -k / x**2
elif best_model == 'Exponential':
    a, b = best_params
    slope = -a * b * np.exp(-b * x)

# Find best model
best_model = min(results, key=lambda x: results[x]['aic'])
print(f"Best model according to AIC: {best_model}")

# Get parameters of the best model
best_params = results[best_model]['params']

def calculate_lambda(x, best_model, best_params):
    if best_model == 'Quadratic':
        a, b, _ = best_params
        return -1*(2 * a * x + b)
    elif best_model == 'Cubic':
        a, b, c, _ = best_params
        return -1*(3 * a * x**2 + 2 * b * x + c)
    elif best_model == 'Inverse':
        k, = best_params
        return k / x**2
    elif best_model == 'Exponential':
        a, b = best_params
        return a * b * np.exp(-b * x)
    else:
        raise ValueError("Unknown model type")
# calculate selected cost
def calculate_selected_cost(alpha, beta, num_samples=10000, threshold=None):
    if threshold is None:
        raise ValueError("Threshold must be provided")

    costs = np.zeros(len(alpha))
    lambda_val = calculate_lambda(x=threshold, best_model=best_model, best_params=best_params)

    non_zero_mask = alpha != 0
    non_zero_alpha = alpha[non_zero_mask]
    non_zero_beta = beta[non_zero_mask]

    if len(non_zero_alpha) > 0:
        samples = np.random.beta(non_zero_alpha, non_zero_beta, size=(num_samples, len(non_zero_alpha)))
        # subject to change
        sample_costs = np.where(samples < threshold, lambda_val * non_zero_alpha, (1 - non_zero_alpha))
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


def adaptive_sampling(data, num_sites, threshold, num_samples=1000, start_seed=10000):
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
                threshold=threshold
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
            selected_sites_year['CipR_Sum_prevalence'] = selected_sites_year['CipR_Sum'] / selected_sites_year['TOTAL']
            selected_sites_year['Threshold'] = threshold
            selected_sites_year['Sample'] = sample + 1
            selected_sites_df = pd.concat([selected_sites_df, selected_sites_year], ignore_index=True)

        all_samples_results[f'sample_{sample + 1}'] = results
        all_samples_selected_sites = pd.concat([all_samples_selected_sites, selected_sites_df], ignore_index=True)

    return {'results': all_samples_results, 'selected_sites': all_samples_selected_sites}

# save the selected sites dataframe based on this strategy
num_sites_list = [1, 5, 10]
thresholds = np.arange(0, 1.02, 0.02)  # Assuming this is your threshold array
total_data = pd.DataFrame()
for num_sites in num_sites_list:
    all_selected_sites = pd.DataFrame()
    print(f"Processing for {num_sites} sites")
    for threshold in thresholds:
        # print(f"  Threshold: {threshold:.2f}")
        sampling_result = adaptive_sampling(df1, num_sites, threshold)
        selected_sites = sampling_result['selected_sites']
        selected_sites['NumSites'] = num_sites
        selected_sites['Threshold'] = threshold
        all_selected_sites = pd.concat([all_selected_sites, selected_sites], ignore_index=True)
    all_selected_sites.to_csv(f'Data/Min Cost {num_sites} sites selected_GISP.csv')
    total_data=pd.concat([total_data,all_selected_sites],ignore_index=True)


def calculate_estimated_prevalence(data):
    # Calculate estimated prevalence for each year, threshold, and number of sites
    prevalence_df = data.groupby(['YEAR', 'Threshold','NumSites']).apply(
        lambda x: pd.Series({
            'TotalSamples': x['TOTAL'].sum(),
            'TotalPositive': x['CipR_Sum'].sum(),
            'EstimatedPrevalence': x['CipR_Sum'].sum() / x['TOTAL'].sum(),
        })
    ).reset_index()

    # Sort the DataFrame by year and threshold
    prevalence_df = prevalence_df.sort_values(['Threshold','NumSites'])

    return prevalence_df
def calculate_stats_pop(prevalence_df):
    # Get unique combinations of prevalence values and NumSites
    prevalence_numsites = prevalence_df[['Threshold', 'NumSites']].drop_duplicates()

    # Initialize the result list
    result = []

    for _, row in prevalence_numsites.iterrows():
        prevalence_value = row['Threshold']
        num_sites = row['NumSites']

        # Filter data for this prevalence value and NumSites
        data = prevalence_df[(prevalence_df['Threshold'] == prevalence_value) &
                             (prevalence_df['NumSites'] == num_sites)].sort_values('YEAR')

        data.reset_index(drop=True, inplace=True)
        dt.reset_index(drop=True, inplace=True)
        data = pd.concat([dt[['YEAR','TOTAL','CipR_Sum']],data[['EstimatedPrevalence']]],axis=1)


        total = data['TOTAL'].sum()
        drug_changed = False
        failure_to_treat = 0
        unnecessary_use = 0

        for _, data_row in data.iterrows():
            if not drug_changed and data_row['EstimatedPrevalence'] >= prevalence_value:
                drug_changed = True

            if drug_changed:
                unnecessary_use += data_row['TOTAL'] - data_row['CipR_Sum']
            else:
                failure_to_treat += data_row['CipR_Sum']

        result.append({
            'Prevalence': prevalence_value,
            'NumSites': num_sites,
            'FailureToTreat': failure_to_treat,
            'UnnecessaryUse': unnecessary_use,
            'Total': total,
            'UnnecessaryUsePercentage': unnecessary_use / total if total > 0 else 0,
            'FailureToTreatPercentage': failure_to_treat / total if total > 0 else 0
        })

    return pd.DataFrame(result)

#calculate auc
def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


# plot the results
def plot_treatment_stats(data):
    def calculate_and_sort_results(num_sites):
        result = calculate_stats_pop(calculate_estimated_prevalence(data))
        result = result[result['NumSites'] == num_sites].sort_values(
            ['FailureToTreatPercentage', 'UnnecessaryUsePercentage'])
        return result

    plt.figure(figsize=(12, 6))
    total_auc = calculate_auc(treatment_stats_sum['FailureToTreatPercentage'],
                              treatment_stats_sum['UnnecessaryUsePercentage'])
    plt.plot(treatment_stats_sum['FailureToTreatPercentage'], treatment_stats_sum['UnnecessaryUsePercentage'],
             color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    for num_sites, color in [(1, 'purple'), (5, 'orange'), (10, 'green')]:
        result = calculate_and_sort_results(num_sites)
        auc = calculate_auc(result['FailureToTreatPercentage'], result['UnnecessaryUsePercentage'])
        plt.plot(result['FailureToTreatPercentage'], result['UnnecessaryUsePercentage'], color=color,
                 label=f'{num_sites} sites (AUC: {auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title('Min Cost Strategy for total population')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Figures/Python Plots/Min Cost Sampling_GISP.png')
    plt.show()

plot_treatment_stats(total_data)