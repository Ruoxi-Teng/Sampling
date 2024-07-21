import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

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

def calculate_lambda(data, threshold, epsilon=1e-10):
    data_sorted = data.sort_values('FailureToTreatPercentage')
    idx = np.searchsorted(data_sorted['FailureToTreatPercentage'], threshold)

    if idx == 0:
        idx = 1
    elif idx == len(data_sorted):
        idx = len(data_sorted) - 1

    point1 = data_sorted.iloc[idx - 1]
    point2 = data_sorted.iloc[idx]

    delta_failure = point2['FailureToTreatPercentage'] - point1['FailureToTreatPercentage']
    delta_unnecessary = point2['UnnecessaryUsePercentage'] - point1['UnnecessaryUsePercentage']

    if abs(delta_failure) < epsilon:
        slope = np.inf if delta_unnecessary > epsilon else -np.inf if delta_unnecessary < -epsilon else 0
    else:
        slope = delta_unnecessary / delta_failure

    lambda_val = -1 / slope if abs(slope) > epsilon else np.inf

    return lambda_val

prevalence_values = np.arange(0, 1.002, 0.002)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)

def calculate_lambda_and_save(data, thresholds):
    results = []
    for threshold in thresholds:
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

        lambda_value = -1 / slope if slope != 0 else np.inf

        results.append({
            'Threshold': threshold,
            'Slope': slope,
            'Lambda': lambda_value
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('Data/lambda_results.csv', index=False)
    return results_df

threshold_values=prevalence_values
lambda_results = calculate_lambda_and_save(treatment_stats_sum, threshold_values)
# Create and save lambda results
lambda_df = pd.DataFrame(lambda_results)
print(lambda_df)
lambda_df.to_csv('Data/lambda_results.csv', index=False)

# Optional: Plot lambda values
plt.figure(figsize=(12, 6))
plt.plot(lambda_df['Threshold'], lambda_df['Lambda'])
plt.title('Lambda Values for Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Lambda')
plt.yscale('symlog')  # Use symlog scale to handle both positive and negative values
plt.grid(True)
plt.savefig('Figures/lambda_values.png')
plt.show()


# Assume your dataframe is called 'df' with columns 'FailureToTreatPercentage' and 'UnnecessaryUsePercentage'
# If not, replace 'df' with your actual dataframe name

# Define the models
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def inverse(x, k):
    return k / x

def exponential(x, a, b):
    return a * np.exp(-b * x)

# Function to calculate AIC
def calculate_aic(n, mse, num_params):
    return 2 * num_params + n * np.log(mse)

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
        results[name] = {'mse': mse, 'aic': aic, 'params': popt}
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

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')

for name, (func, _) in models.items():
    if name in results:
        x_plot = np.linspace(min(x), max(x), 100)
        y_plot = func(x_plot, *results[name]['params'])
        plt.plot(x_plot, y_plot, label=f'{name} fit')

plt.xlabel('Failure to Treat Percentage')
plt.ylabel('Unnecessary Use Percentage')
plt.legend()
plt.title('Model Fitting Comparison')
plt.savefig('Figures/model_fitting_comparison.png')
plt.show() # the cubic shows the best fitting result

# define a new function to calculate lambda

def calculate_lambda_fitting(data,threshold):
    if threshold<=0.16:
        slope=-241.31644717*threshold**2*3+87.2796778*threshold*2-12.25339056
    else:
        slope= 0
    lambda_value = -1 / slope if slope != 0 else np.inf
    return lambda_value


def calculate_lambda_and_save_fitting(data, thresholds):
    results = []
    for threshold in thresholds:
        if threshold <= 0.16:
            slope = -241.31644717 * threshold ** 2 * 3 + 87.2796778 * threshold * 2 - 12.25339056
        else:
            slope = 0
        lambda_value = -1 / slope if slope != 0 else np.inf
        results.append({
            'Threshold': threshold,
            'Slope': slope,
            'Lambda': lambda_value
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('Data/lambda_results_fitting.csv', index=False)
    return results_df

threshold_values = np.arange(0, 0.36, 0.002)
lambda_results_fitting = calculate_lambda_and_save_fitting(treatment_stats_sum, threshold_values)
# Create and save lambda results
lambda_df_fitting = pd.DataFrame(lambda_results_fitting)
print(lambda_df_fitting)
lambda_df_fitting.to_csv('Data/lambda_results_fitting.csv', index=False)

# Optional: Plot lambda values
plt.figure(figsize=(12, 6))
plt.plot(lambda_df_fitting['Threshold'], lambda_df_fitting['Lambda'])
plt.title('Lambda Values for Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Lambda')
plt.yscale('symlog')  # Use symlog scale to handle both positive and negative values
plt.grid(True)
plt.savefig('Figures/lambda_values_fitting.png')
plt.show()
