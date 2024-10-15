import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import uniform

# Load and preprocess data
df = pd.read_csv("Data/CIP_summary_by_sites.csv")
df = df.drop(columns=df.columns[0])
df['CipR_Sum_prevalence'] = df['CipR_Sum'] / df['TOTAL']
dt = pd.read_csv("Data/CIP_summary.csv")
dt = dt.drop(columns=dt.columns[0])

#plot prevalence trend
dt.plot(x='YEAR', y='CipR_Sum_prevalence', figsize=(10,6))

plt.xlabel("Year")
plt.ylabel("Antibiotic Resistence Prevalence")
plt.legend()
plt.title('Cip Antibiotic Resistence Prevalence over years')
plt.savefig('Figures/Prevalence_Cip_GISP.png')
plt.show()

clinics = df['CLINIC'].unique()

# Create a color map
color_map = plt.cm.get_cmap('tab20')  # You can change 'tab20' to other colormaps

# Create the plot
plt.figure(figsize=(12, 6))

for i, clinic in enumerate(clinics):
    clinic_data = df[df['CLINIC'] == clinic]
    plt.plot(clinic_data['YEAR'], clinic_data['CipR_Sum_prevalence'],
             label=clinic, color=color_map(i / len(clinics)))

plt.xlabel("Year")
plt.ylabel("Antibiotic Resistence Prevalence")
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=10)
# Adjust the layout to make room for the legend
plt.tight_layout()
plt.title('Cip Antibiotic Resistence Prevalence across clinics over years')
plt.savefig('Figures/Prevalence_Cip_clinics_GISP.png')
plt.show()

# model fitting
def Expo(x, A):
    y = np.exp(x*(A))-1
    return y

dt['YEAR']=np.arange(0,23,1)

parameters, covariance = curve_fit(Expo, dt['YEAR'], dt['CipR_Sum_prevalence'])
fit_A = parameters[0]

#plot comparison
fit_y = Expo(dt['YEAR'], fit_A)
plt.plot(dt['YEAR'],  dt['CipR_Sum_prevalence'], 'o', label='data')
plt.plot(dt['YEAR'], fit_y, '-', label='fit')
plt.legend()

dt1=df[df['YEAR']==2022]
min=dt1['CipR_Sum_prevalence'].min()
max=dt1['CipR_Sum_prevalence'].max()

# Set random seed for reproducibility
np.random.seed(573)

# Parameters
n_sites = 100
n_residents = 10000
n_years = 23  # 0 to 22
beta_max,beta_min = np.log(1+max)/22, np.log(1+min)/22  # example range, adjust as needed

# Generate betas for each site
betas = uniform.rvs(loc=beta_min, scale=beta_max-beta_min, size=n_sites)

# Generate clinic names
clinics = [f'CLINIC_{i:03d}' for i in range(n_sites)]

# Simulation function
def simulate_prevalence(beta, years):
    return np.exp(beta * years)-1

# Generate data for each site
data = []
for clinic, beta in zip(clinics, betas):
    years = np.arange(n_years)
    prevalence = simulate_prevalence(beta, years)
    for year, prev in zip(years, prevalence):
        total = n_residents
        positive = int(prev * total)
        data.append({
            'CLINIC': clinic,
            'YEAR': 2000 + year,
            'TOTAL': total,
            'CipR_Sum': positive,
            'CipR_Sum_prevalence': prev
        })

# Create the first dataframe
df1 = pd.DataFrame(data)

df2 = df1.groupby('YEAR').agg({
    'TOTAL': 'sum',
    'CipR_Sum': 'sum'
}).reset_index()

df2['CipR_Sum_prevalence'] = df2['CipR_Sum'] / df2['TOTAL']

# Display the first few rows of each dataframe
print(df1.head())
print("\n")
print(df2.head())

# Save dataframes to CSV
df1.to_csv('Data/site_prevalence_data.csv', index=False)
df2.to_csv('Data/total_prevalence_data.csv', index=False)

clinics = df1['CLINIC'].unique()
# Assuming df1 contains simulated data and data contains actual data

# Get unique clinics from both datasets
clinics_simulated = df1['CLINIC'].unique()
clinics_actual = df['CLINIC']
# Create a color map
color_map = plt.cm.get_cmap('tab20')

# Create the plot
plt.figure(figsize=(12, 6))

for i, clinic in enumerate(clinics):
    clinic_data = df1[df1['CLINIC'] == clinic]
    plt.plot(clinic_data['YEAR'], clinic_data['CipR_Sum_prevalence'],
             label=clinic, color=color_map(i / len(clinics)))

plt.xlabel("Year")
plt.ylabel("Antibiotic Resistence Prevalence")
# plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=10)
# Adjust the layout to make room for the legend
plt.tight_layout()
plt.title('Cip Antibiotic Resistence Prevalence across clinics over years Simulated')
plt.savefig('Figures/Prevalence_Cip_clinics_simulated.png')
plt.show()

# Create the plot
plt.figure(figsize=(15, 8))

# Plot simulated data (df1) in light grey
for i, clinic in enumerate(clinics_simulated):
    clinic_data = df1[df1['CLINIC'] == clinic]
    plt.plot(clinic_data['YEAR'], clinic_data['CipR_Sum_prevalence'],
             color='lightgrey', alpha=0.5, linewidth=1)

# Plot actual data with colors
for i, clinic in enumerate(clinics_actual):
    clinic_data = df[df['CLINIC'] == clinic]
    plt.plot(clinic_data['YEAR'], clinic_data['CipR_Sum_prevalence'],
             label=clinic, color=color_map(i / len(clinics_actual)), linewidth=2)

plt.xlabel("Year")
plt.ylabel("Antibiotic Resistance Prevalence")
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5)
plt.tight_layout()
plt.title('Cip Antibiotic Resistance Prevalence: Actual vs Simulated')

# Add a text annotation to explain the grey lines
plt.text(0.02, 0.98, 'Grey lines: Simulated data', transform=plt.gca().transAxes,
         verticalalignment='top', fontsize=10, alpha=0.7)

# Adjust the layout to make room for the legend
plt.subplots_adjust(bottom=0.2)

# Save the figure
plt.savefig('Figures/Prevalence_Cip_clinics_actual_vs_simulated.png', dpi=300, bbox_inches='tight')

plt.show()