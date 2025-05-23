import pandas as pd
import numpy as np

# Read the clustered data
print("Reading clustered data...")
df = pd.read_csv('churn_data_clustered_sample.csv')

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

# Save the summary to CSV in the same folder
output_path = 'cluster_summary.csv'
cluster_summary.to_csv(output_path)
print(f"\nCluster summary saved to '{output_path}'") 