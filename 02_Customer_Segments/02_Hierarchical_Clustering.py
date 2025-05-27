import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import os
from tqdm import tqdm
import gc  # For garbage collection

print("Starting the analysis...")

# Set style for better visualizations
plt.style.use('default')
sns.set_theme()

# Create diagrams folder if it doesn't exist
diagrams_folder = os.path.join(os.path.dirname(__file__), 'diagrams')
if not os.path.exists(diagrams_folder):
    os.makedirs(diagrams_folder)

# Define the file path
file_path = '01_Initial_Data/Churn_Data_cleaned.csv'

# First, read just the header to get column names
print("Reading column names...")
df_columns = pd.read_csv(file_path, nrows=0)
print("Available columns:", df_columns.columns.tolist())

# Read numerical and boolean columns, excluding 'churn'
print("Reading the data...")
# Read a sample of 5000 rows
df = pd.read_csv(file_path, 
                 usecols=lambda x: (pd.api.types.is_numeric_dtype(pd.read_csv(file_path, usecols=[x]).dtypes[x]) or 
                                  pd.api.types.is_bool_dtype(pd.read_csv(file_path, usecols=[x]).dtypes[x])) and 
                                  x not in ['churn', 'Customer_ID'],
                 nrows=5000)  # Only read 5000 rows

print("Data loaded. Shape:", df.shape)
print("Columns used:", df.columns.tolist())
print("\nColumn types:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

# Convert to float32 to reduce memory usage
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
    elif df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')
    elif df[col].dtype == 'bool':
        df[col] = df[col].astype('int32')  # Convert boolean to int for clustering

# Handle missing values
print("Handling missing values...")
df = df.fillna(df.mean())

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Free up memory
del df
gc.collect()

# Perform hierarchical clustering with a subset of data for the dendrogram
print("Creating dendrogram...")
sample_size = min(1000, len(X_scaled))  # Use a smaller sample for the dendrogram
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

linkage_matrix = linkage(X_sample, method='ward')

# Create a figure for the dendrogram with better visualization
plt.figure(figsize=(15, 10))
dend = dendrogram(linkage_matrix,
                 truncate_mode='level',
                 p=5,
                 leaf_font_size=10,
                 leaf_rotation=90)

# Add a title and labels
plt.title('Customer Segmentation Dendrogram\n(Shows how customers are grouped based on their characteristics)', 
         fontsize=14, pad=20)
plt.xlabel('Customer Index', fontsize=12)
plt.ylabel('Distance (Dissimilarity)', fontsize=12)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(diagrams_folder, 'hierarchical_clustering_dendrogram.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis of the hierarchical structure:")
print("1. The dendrogram shows the complete hierarchical relationship between customers")
print("2. The height of each merge shows the distance between clusters being merged")
print("3. Longer vertical lines indicate more distinct separations between groups")
print("4. You can analyze the relationships at any level of the hierarchy")
print("\nThe dendrogram has been saved to 'hierarchical_clustering_dendrogram.png'")
print("You can use this to identify natural groupings in your customer base.")

# Calculate and display some basic statistics about the hierarchical structure
print("\nHierarchical Structure Statistics:")
print(f"Total number of merges: {len(linkage_matrix)}")
print(f"Maximum merge distance: {linkage_matrix[-1, 2]:.2f}")
print(f"Average merge distance: {np.mean(linkage_matrix[:, 2]):.2f}")

# Analyze churn distribution in the sample
print("\nAnalyzing churn distribution in the sample dataset...")
# Read the churn column for the sample
churn_data = pd.read_csv(file_path, usecols=['churn'], nrows=5000)
churn_distribution = churn_data['churn'].value_counts()
churn_percentage = (churn_distribution / len(churn_data) * 100).round(2)

print("\nChurn Distribution:")
print(f"Total customers in sample: {len(churn_data)}")
print("\nCounts:")
print(churn_distribution)
print("\nPercentages:")
print(churn_percentage)

# Create a bar plot of churn distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=churn_data, x='churn')
plt.title('Distribution of Churn in Sample Dataset', fontsize=14, pad=20)
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig(os.path.join(diagrams_folder, 'churn_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Check cluster sizes
cluster_sizes = pd.Series(cluster_labels).value_counts()
print("\nInitial cluster sizes:")
print(cluster_sizes)

# If any cluster is too small, increase number of clusters and try again
while any(cluster_sizes < 50):  # Assuming a default minimum cluster size
    n_clusters += 1
    print(f"\nSome clusters too small. Trying with {n_clusters} clusters...")
    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(X_scaled)
    cluster_sizes = pd.Series(cluster_labels).value_counts()
    print("New cluster sizes:")
    print(cluster_sizes)
    
    if n_clusters > 10:  # Stop if we can't find a good solution
        print("Warning: Could not find clusters with minimum size requirement")
        break

# Free up memory
del X_scaled
gc.collect()

# Read the original data again to add cluster labels
print("Adding cluster labels to the original data...")
df = pd.read_csv(file_path, nrows=5000)  # Read the same sample
df['Cluster'] = cluster_labels

# Calculate cluster statistics
print("Calculating cluster statistics...")
# Include both numeric and boolean columns in cluster statistics
numeric_cols = [col for col in df.select_dtypes(include=[np.number, 'bool']).columns if col != 'churn']
cluster_stats = df.groupby('Cluster')[numeric_cols].mean()
print("\nCluster Statistics:")
print(cluster_stats)

# Create a detailed summary table for each cluster
print("\nCreating detailed cluster summary...")
cluster_summary = pd.DataFrame()

# Basic cluster information
cluster_summary['Number_of_Customers'] = df['Cluster'].value_counts().sort_index()
cluster_summary['Percentage_of_Total'] = (df['Cluster'].value_counts().sort_index() / len(df) * 100).round(2)

# Key metrics
cluster_summary['Average_Revenue'] = df.groupby('Cluster')['rev_Mean'].mean().round(2)
cluster_summary['Average_Completed_Calls'] = df.groupby('Cluster')['complete_Mean'].mean().round(2)
cluster_summary['Average_Months_in_Service'] = df.groupby('Cluster')['months'].mean().round(2)
cluster_summary['Average_Blocked_Data_Calls'] = df.groupby('Cluster')['blck_dat_Mean'].mean().round(2)
cluster_summary['Average_Customer_Care_Minutes'] = df.groupby('Cluster')['ccrndmou_Mean'].mean().round(2)

# Additional insightful metrics
cluster_summary['Average_Minutes_of_Use'] = df.groupby('Cluster')['mou_Mean'].mean().round(2)
cluster_summary['Average_Revenue_per_Month'] = (df.groupby('Cluster')['rev_Mean'].mean() / df.groupby('Cluster')['months'].mean()).round(2)
cluster_summary['Average_Call_Drop_Rate'] = df.groupby('Cluster')['drop_vce_Mean'].mean().round(2)

# Format the summary table
print("\nDetailed Cluster Summary:")
print(cluster_summary)

# Save the summary to CSV
cluster_summary.to_csv('cluster_summary.csv')
print("\nCluster summary saved to 'cluster_summary.csv'")

# Save the clustered data
print("Saving results...")
df.to_csv('churn_data_clustered_sample.csv', index=False)

# Create a heatmap of cluster centers with better visualization
plt.figure(figsize=(15, 10))
sns.heatmap(cluster_stats, 
            annot=True, 
            cmap='YlOrRd', 
            fmt='.2f',
            cbar_kws={'label': 'Mean Value'})
plt.title('Cluster Characteristics\n(Average values for each feature in each cluster)', 
         fontsize=14, pad=20)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(diagrams_folder, 'cluster_centers_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print cluster sizes
cluster_sizes = df['Cluster'].value_counts().sort_index()
print("\nCluster Sizes:")
print(cluster_sizes)

print("Analysis complete!")