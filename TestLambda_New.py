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

prevalence_values = np.arange(0, 1.002, 0.002)
treatment_stats_sum = calculate_treatment_stats_sum(dt, prevalence_values)

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

# Calculate lambda
lambda_values = np.where(slope != 0, -1 / slope, np.inf)

# Create and save the results dataframe
results_df = pd.DataFrame({
    'FailureToTreatPercentage': x,
    'UnnecessaryUsePercentage': y,
    'FittedValue': fitted_values,
    'Slope': slope,
    'Lambda': lambda_values
})

results_df.to_csv('Data/fitting_results.csv', index=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')

for name, res in results.items():
    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = res['func'](x_plot, *res['params'])
    plt.plot(x_plot, y_plot, label=f'{name} fit')

plt.xlabel('Failure to Treat Percentage')
plt.ylabel('Unnecessary Use Percentage')
plt.legend()
plt.title('Model Fitting Comparison')
plt.savefig('Figures/model_fitting_comparison.png')
plt.show()

print("Results saved to 'Data/fitting_results.csv'")