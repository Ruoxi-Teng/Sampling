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

# Usage
prevalence_values = np.arange(0, 1.02, 0.02)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)
treatment_stats_customized = calculate_treatment_stats_sites(df, prevalence_values)
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


#calculate auc
def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


def calculate_treatment_stats_min_cost(all_sites, selected_sites_samples, prevalence_values):
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
            failure_to_treat = 0
            unnecessary_use = 0
            total = 0

            for clinic in all_clinics:
                clinic_data = all_sites[all_sites['CLINIC'] == clinic]
                total += clinic_data['TOTAL'].sum()

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
                            unnecessary_use += year_clinic['TOTAL'].sum() - year_clinic['CipR_Sum'].sum()
                        else:
                            failure_to_treat += year_clinic['CipR_Sum'].sum()
                    else:
                        # For unselected clinics, base decision on estimated prevalence from selected sites
                        if not year_selected.empty and not year_clinic.empty:
                            estimated_prevalence = year_selected['CipR_Sum'].sum() / year_selected['TOTAL'].sum()
                            if not drug_changed and estimated_prevalence >= prevalence_value:
                                drug_changed = True
                            if drug_changed:
                                unnecessary_use += year_clinic['TOTAL'].sum() - year_clinic['CipR_Sum'].sum()
                            else:
                                failure_to_treat += year_clinic['CipR_Sum'].sum()

            all_results.append({
                'Sample': sample_id,
                'Prevalence': prevalence_value,
                'FailureToTreat': failure_to_treat,
                'UnnecessaryUse': unnecessary_use,
                'Total': total,
                'UnnecessaryUsePercentage': unnecessary_use / total if total > 0 else 0,
                'FailureToTreatPercentage': failure_to_treat / total if total > 0 else 0
            })

    # Convert all results to DataFrame
    final_results = pd.DataFrame(all_results)

    return final_results


# plot random sampling result
def plot_min_cost_customized_results(results, treatment_stats_sum, treatment_stats_customized, num_clinics):
    plt.figure(figsize=(12, 6), dpi=700)

    # plot each sample curve
    sample_final = calculate_treatment_stats_min_cost(df, results, prevalence_values)
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
        f'Failure to Treat v.s. Unnecessary Treatment under different threshold (2000-2022): Min Cost {num_clinics} sites samples')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf()



# result: min cost sampling
for num_clinics in [1, 5, 10]:
    min_cost_results = pd.read_csv(f"Data/Min Cost {num_clinics} sites selected_GISP.csv")
    plot = plot_min_cost_customized_results(min_cost_results,treatment_stats_sum,treatment_stats_customized,num_clinics)
    plot.savefig(f'Figures/Min Cost {num_clinics} sites Customized_GISP.png')
    plt.show()

