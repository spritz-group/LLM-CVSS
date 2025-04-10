import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

df = pd.read_parquet('../../dataset.parquet')

results = {}

for column in ['AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A']:
    print(f"Value counts for column '{column}':")
    value_counts = df[column].value_counts()
    print(value_counts)
    results[column] = (value_counts / value_counts.sum()).to_dict()

# Prepare data 
categories = list(results.keys())
unique_values = list(set(val for counts in results.values() for val in counts.keys()))
plot_data = pd.DataFrame(index=unique_values, columns=categories).fillna(0)

for column, counts in results.items():
    for value, count in counts.items():
        plot_data.at[value, column] = count

# Plot 
ax = plot_data.T.plot(kind='bar', stacked=True, figsize=(10, 6), zorder=5)

# Add labels inside the bars
for container, value_label in zip(ax.containers, unique_values):
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                value_label,
                ha='center',
                va='center',
                fontsize=14,
                color='white',
                zorder=10
            )

#plt.title('Stacked Bar Plot of Dataset Distribution')
plt.xlabel('Elements')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.ylim(0, 1)
ax.legend_.remove()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
plt.tight_layout()
plt.savefig('dataset_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()
