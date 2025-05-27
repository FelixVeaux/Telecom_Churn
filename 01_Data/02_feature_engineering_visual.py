import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Create sample data to demonstrate the transformation
np.random.seed(42)
n_samples = 10

# Original data with missing values
original_data = pd.DataFrame({
    'mailflag': [np.nan, 'Y', np.nan, np.nan, 'Y', np.nan, np.nan, 'Y', np.nan, np.nan],
    'last_swap': [np.nan, np.nan, 'Y', np.nan, np.nan, 'Y', np.nan, np.nan, 'Y', np.nan],
    'tot_ret': ['Y', np.nan, np.nan, 'Y', np.nan, np.nan, 'Y', np.nan, np.nan, 'Y']
})

# Create binary features (no missing values)
binary_data = pd.DataFrame({
    'mail': (~original_data['mailflag'].isna()).astype(int),
    'phoneswap': (~original_data['last_swap'].isna()).astype(int),
    'retentioncall': (~original_data['tot_ret'].isna()).astype(int)
})

# Create figure with more width to accommodate legends
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot original data (missing=red, present=green)
sns.heatmap(original_data.isna(), cmap=["#2ecc71", "#e74c3c"],
            cbar=False, ax=axes[0], yticklabels=range(1, n_samples + 1))
axes[0].set_title('Original Data with Missing Values', pad=10, fontsize=13)
axes[0].set_xticklabels(['Mail Flag', 'Last Swap', 'Retention Call'], rotation=0)
axes[0].set_ylabel('Sample')
axes[0].set_xlabel('Features')

# Create legend for original data
legend1 = axes[0].legend(handles=[
    plt.Line2D([0], [0], marker='s', color='w', label='Present', markerfacecolor='#2ecc71', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', label='Missing', markerfacecolor='#e74c3c', markersize=10)
], loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, title='Legend')

# Plot binary features (0=blue, 1=green)
sns.heatmap(binary_data, cmap=["#3498db", "#2ecc71"],
            cbar=False, ax=axes[1], yticklabels=range(1, n_samples + 1), vmin=0, vmax=1)
axes[1].set_title('Binary Features Created from Missing Values', pad=10, fontsize=13)
axes[1].set_xticklabels(['Mail', 'Phone Swap', 'Retention Call'], rotation=0)
axes[1].set_ylabel('Sample')
axes[1].set_xlabel('Features')

# Create legend for binary features
legend2 = axes[1].legend(handles=[
    plt.Line2D([0], [0], marker='s', color='w', label='0 (Not Present)', markerfacecolor='#3498db', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', label='1 (Present)', markerfacecolor='#2ecc71', markersize=10)
], loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, title='Legend')

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.subplots_adjust(right=0.85)  # Make room for legends
plt.savefig('feature_engineering_visual.png', dpi=300, bbox_inches='tight')
plt.close() 