import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def preprocess_data(data):
    min_year, max_year = data['YEAR'].min(), data['YEAR'].max()
    total_years = max_year - min_year + 1
    site_counts = data.groupby('CLINIC')['YEAR'].nunique()
    sites_with_all_years = site_counts[site_counts == total_years].index
    return data[data['CLINIC'].isin(sites_with_all_years)]


def calculate_treatment_stats_sum(data, prevalence_values):
    total = data['TOTAL'].sum()
    drug_change = (data['CipRsum_prevalence'].values[:, np.newaxis] >= prevalence_values).astype(int)
    failure_to_treat = ((1 - drug_change) * data['CipRsum'].values[:, np.newaxis]).sum(axis=0)
    unnecessary_use = (drug_change * (data['TOTAL'].values[:, np.newaxis] - data['CipRsum'].values[:, np.newaxis])).sum(
        axis=0)

    return pd.DataFrame({
        'Prevalence': prevalence_values,
        'FailureToTreat': failure_to_treat,
        'UnnecessaryUse': unnecessary_use,
        'Total': total,
        'UnnecessaryUsePercentage': unnecessary_use / total,
        'FailureToTreatPercentage': failure_to_treat / total
    })


def generate_multiple_random_samples(df, num_clinics, num_samples, prevalence_values):
    all_results = []
    selected_clinics_data = []

    for i in range(num_samples):
        np.random.seed(573 + i * num_clinics)
        unique_values = df['CLINIC'].unique()
        selected_clinics = np.random.choice(unique_values, num_clinics, replace=False)

        subset_df = df[df['CLINIC'].isin(selected_clinics)].drop(columns=['CLINIC'])
        subset_df_agg = subset_df.groupby('YEAR').sum().reset_index()
        subset_df_agg['CipRsum_prevalence'] = subset_df_agg['CipRsum'] / subset_df_agg['TOTAL']

        dt5 = dt.copy()
        dt5['CipRsum_prevalence'] = subset_df_agg['CipRsum_prevalence'].values
        treatment_stats = calculate_treatment_stats_sum(dt5, prevalence_values)
        treatment_stats['Sample'] = i + 1
        all_results.append(treatment_stats)

        # Store selected clinics information
        selected_clinics_data.append({
            'Sample': i + 1,
            'Selected_Clinics': ', '.join(selected_clinics)
        })

    # Create a dataframe with selected clinics information
    selected_clinics_df = pd.DataFrame(selected_clinics_data)

    return pd.concat(all_results, ignore_index=True), selected_clinics_df


def calculate_auc(x, y):
    sorted_data = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_data)
    auc = trapz(y_sorted, x_sorted)
    return auc / (max(x_sorted) * max(y_sorted))


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def calculate_aic(n, mse, num_params):
    return 2 * num_params + n * np.log(mse)


def fit_models(x, y):
    models = {
        'Quadratic': (quadratic, 3),
        'Cubic': (cubic, 4)
    }

    results = {}
    for name, (func, num_params) in models.items():
        try:
            popt, _ = curve_fit(func, x, y)
            y_pred = func(x, *popt)
            mse = np.mean((y - y_pred) ** 2)
            aic = calculate_aic(len(x), mse, num_params)
            r2 = r2_score(y, y_pred)
            results[name] = {'popt': popt, 'aic': aic, 'r2': r2, 'func': func}
        except:
            print(f"Fitting failed for {name}")

    return results


def plot_treatment_paradigms(results, total_stats, num_clinics):
    plt.figure(figsize=(12, 6), dpi=500)

    for sample in results['Sample'].unique():
        sample_data = results[results['Sample'] == sample]
        plt.plot(sample_data['FailureToTreatPercentage'], sample_data['UnnecessaryUsePercentage'], color='gold',
                 alpha=0.01)

    x = results['FailureToTreatPercentage'].values
    y = results['UnnecessaryUsePercentage'].values
    fit_results = fit_models(x, y)

    best_model = min(fit_results, key=lambda k: fit_results[k]['aic'])
    best_func = fit_results[best_model]['func']
    best_popt = fit_results[best_model]['popt']
    best_r2 = fit_results[best_model]['r2']

    x_smooth = np.linspace(min(x), max(x), 1000)
    y_smooth = best_func(x_smooth, *best_popt)
    plt.plot(x_smooth, y_smooth, color='red',
             label=f'Best fit ({best_model}, R² = {best_r2:.4f})')

    total_auc = calculate_auc(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'])
    plt.plot(total_stats['FailureToTreatPercentage'], total_stats['UnnecessaryUsePercentage'], color='blue',
             label=f'All sites (AUC: {total_auc:.4f})')

    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Treatment paradigms under different cutoff value (2000-2022): Fixed {num_clinics} sites samples')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True)
    return plt.gcf(), fit_results


# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(columns=df.columns[0])
df['CipRsum_prevalence'] = df['CipRsum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(columns=dt.columns[0])

df1 = preprocess_data(df)
dt1 = df1[df1.columns[1:4]].groupby('YEAR').sum().reset_index()
dt1['CipRsum_prevalence'] = dt1['CipRsum'] / dt1['TOTAL']

df1 = preprocess_data(df)
prevalence_values = np.arange(0, 1.000, 0.002)
total_stats = calculate_treatment_stats_sum(dt, prevalence_values)

# Generate and plot results for 1, 5, and 10 clinics
for num_clinics in [1, 5, 10]:
    random_sample_results, selected_clinics_df = generate_multiple_random_samples(df1, num_clinics, 1500,
                                                                                  prevalence_values)
    random_sample_results.to_csv(f"Data/fixed_summary_{num_clinics}.csv", index=False)

    plot, fit_results = plot_treatment_paradigms(random_sample_results, total_stats, num_clinics)
    plot.savefig(f'Figures/Fixed {num_clinics} sites_with_fit.png')
    plt.close()

    # Plot comparing different fits
    plt.figure(figsize=(12, 6), dpi=500)
    x = random_sample_results['FailureToTreatPercentage'].values
    y = random_sample_results['UnnecessaryUsePercentage'].values
    x_smooth = np.linspace(min(x), max(x), 1000)

    for model, result in fit_results.items():
        y_smooth = result['func'](x_smooth, *result['popt'])
        plt.plot(x_smooth, y_smooth, label=f'{model} (AIC: {result["aic"]:.2f}, R²: {result["r2"]:.4f})')

    plt.scatter(x, y, color='gray', alpha=0.1, s=1)
    plt.xlabel('Failure to Treat (%)')
    plt.ylabel('Unnecessary Treatment (%)')
    plt.title(f'Comparison of Fits for {num_clinics} Clinics')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Figures/Fit_comparison_{num_clinics}_clinics_fixed.png')
    plt.close()

    print(f"\nFit results for {num_clinics} clinics:")
    for model, result in fit_results.items():
        print(f"{model}: AIC = {result['aic']:.2f}, R² = {result['r2']:.4f}")