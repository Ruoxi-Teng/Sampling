import pandas as pd
import numpy as np
dt = pd.read_excel(r"Data/NEEMA_GISP Table 1_Sex Bx_Antibiotics_2000_2022_20240318.xlsx")

# Replace '.' with 0
dt = dt.replace('.', 0)

# Keep a copy of the original data
original = dt.copy()

# Convert columns 5 to 29 to numeric
dt.iloc[:, 4:29] = dt.iloc[:, 4:29].apply(pd.to_numeric)

# Drop the first and fourth columns
dt = dt.drop(dt.columns[[0, 3]], axis=1)

# Aggregate by clinic and year
dt1 = dt.groupby(['CLINIC', 'YEAR']).sum().reset_index()

# Drop the 'CLINIC' column
dt1 = dt1.drop('CLINIC', axis=1)

# Further calculate sum of each year
dt2 = dt1.groupby('YEAR').sum().reset_index()


# Copy dt2 to dt3
dt3 = dt2.copy()

# Sum of each prevalence when includes certain drug
# Rough method
dt3['TetRsum'] = dt3.filter(like='TetR').sum(axis=1)
dt3['PenRsum'] = dt3.filter(like='PenR').sum(axis=1)
dt3['AziRsum'] = dt3.filter(like='AziR').sum(axis=1)
dt3['CipRsum'] = dt3.filter(like='CipR').sum(axis=1)
dt3['CfxRsum'] = dt3.filter(like='CfxR').sum(axis=1)
dt3['CroRsum'] = dt3.filter(like='CroR').sum(axis=1)

# Calculate prevalence
for col in dt3.columns[2:]:  # Start from the third column
    new_col_name = f"{col}_prevalence"
    dt3[new_col_name] = dt3[col] / dt3.iloc[:, 1]  # Divide by the second column (index 1)

# Select specific columns
dt4 = dt3[["YEAR", "TetRsum_prevalence", "PenRsum_prevalence", "AziRsum_prevalence",
           "CipRsum_prevalence", "CfxRsum_prevalence", "CroRsum_prevalence"]]

# Rename columns
dt4.columns = ["Year", "TET", "PEN", "AZI", "CIP", "CFX", "CRO"]

# Select specific columns
dt5 = dt3[["YEAR", "TOTAL", "CipRsum", "CipRsum_prevalence"]]

# generate by sites data
# Drop the first column
dt_by_year_sites = original.iloc[:, 1:]

# Drop the third column (now the second after previous drop)
dt_by_year_sites = dt_by_year_sites.drop(dt_by_year_sites.columns[2], axis=1)

# Convert columns 2 to 27 to numeric
dt_by_year_sites.iloc[:, 1:27] = dt_by_year_sites.iloc[:, 1:27].apply(pd.to_numeric)

# Group by CLINIC and YEAR, and sum other columns
dt_by_year_sites = dt_by_year_sites.groupby(['CLINIC', 'YEAR']).sum().reset_index()

# Save CLINIC column
clinic = dt_by_year_sites['CLINIC']

# Drop CLINIC column
dt_by_year_sites = dt_by_year_sites.drop('CLINIC', axis=1)

# Sum across anti-resistance
for resistance in ['TetR', 'PenR', 'AziR', 'CipR', 'CfxR', 'CroR']:
    dt_by_year_sites[f'{resistance}sum'] = dt_by_year_sites.filter(like=resistance).sum(axis=1)

# Select the columns we are interested in
columns_of_interest = ['YEAR', 'TOTAL', 'TetRsum', 'PenRsum', 'AziRsum', 'CipRsum', 'CfxRsum', 'CroRsum']
dt_by_year_sites = dt_by_year_sites[columns_of_interest]

# Add clinic column back
dt_by_year_sites = pd.concat([clinic, dt_by_year_sites], axis=1)

# Create processed_data_by_sites
processed_data_by_sites = dt_by_year_sites.copy()

# Create dt_by_year_sites_cip
dt_by_year_sites_cip = dt_by_year_sites[['CLINIC', 'YEAR', 'TOTAL', 'CipRsum']]

# print(dt_by_year_sites_cip)

dt_by_year_sites_cip.to_csv('Data/CIP_summary_by_sites.csv')

dt4.to_csv("Data/Total_summary.csv")

dt5.to_csv("Data/CIP_summary.csv")