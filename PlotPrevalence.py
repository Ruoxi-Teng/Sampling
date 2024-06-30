import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Data/Total_summary.csv")
# Create the line graph
plt.figure(figsize=(12, 6))
for column in df.columns[2:]:  # Exclude the 'Year' column
    plt.plot(df['Year'], df[column], label=column)

plt.xlabel('Year')
plt.ylabel('Prevalence')
plt.title('Antimicrobial resistance changing by Year')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent cutoff of labels
plt.tight_layout()


# Save the plot as 'prevalence.png' in the 'figure' folder
plt.savefig('Figures/prevalence.png')

# Show the plot
plt.show()