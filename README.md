# Telecom Customer Churn Analysis Framework

## Project Overview

Customer churn represents a significant challenge in the telecom industry, where acquiring new customers costs 5-25 times more than retaining existing ones. This repository provides a comprehensive workflow for telecom companies to analyze customer churn patterns, predict at-risk customers, and strategically allocate promotion budgets to maximize retention.

This framework serves:
- **Data Scientists**: For implementing predictive models and survival analysis
- **Business Analysts**: For customer segmentation and insights generation
- **Marketing Teams**: For targeted promotion planning and budget optimization

## Directory & File Structure

### 01_Initial_Data/
Contains raw data and initial processing scripts:
- `01_Data_Cleaning.ipynb` - Main data cleaning and preprocessing notebook
- `feature_engineering_visual.py` - Script for generating feature engineering visualizations
- `Churn_Data_cleaned.csv` - Cleaned dataset after preprocessing
- `Cell1.csv` - Raw telecom data
- `data_documentation_class.xls` - Data dictionary and documentation
- `figures/` - Directory containing generated visualizations

### 02_Customer_Segments/
Houses customer segmentation analysis:
- `01_Customer_Clustering.ipynb` - Main clustering analysis notebook
- `02_Hierarchical_Clustering.py` - Implementation of hierarchical clustering
- `create_cluster_summary.py` - Script for generating cluster statistics
- `Churn_Data_with_Clusters.csv` - Dataset with cluster assignments
- `cluster_summary.csv` - Summary statistics of clusters
- `hierarchical_clustering_dendrogram.png` - Visualization of cluster hierarchy
- `cluster_centers_heatmap.png` - Heatmap of cluster characteristics
- `diagrams/` - Directory containing additional visualizations

### 03_KMF/
Kaplan-Meier and Cox Proportional Hazards analysis:
- `01_KMF_VariableSelection.ipynb` - Variable selection for survival analysis
- `02_KMF_COX.ipynb` - Implementation of KMF and Cox models
- `03_KMF_COX_CHURN_Prediction.ipynb` - Churn prediction using survival analysis
- `model_comparison.txt` - Comparison of different model performances
- `column_analysis_cleaneddf.csv` - Analysis of cleaned dataset columns
- `figures/` - Directory containing survival analysis visualizations

### 04_Chrun_Prediction/
Churn prediction models:
- `01_Chrun_Prediction.ipynb` - Implementation of churn prediction models

### 05_RFM_Analysis/
RFM (Recency, Frequency, Monetary) analysis:
- `Telecom_RFM_segmentation.ipynb` - RFM analysis and customer segmentation
- `rfm_variable_candidates.xlsx` - Potential RFM variables for analysis

### Top-Level Files
- `feature_engineering_visual.png` - Visualization of feature engineering process
- `churn_data_clustered_sample.csv` - Sample dataset with clustering results
- `cluster_summary.csv` - Summary statistics of customer clusters
- `final_clean_data_dictionary.xlsx` - Comprehensive data dictionary
- `KMF_COX_CHURN_Prediction.ipynb` - Main prediction notebook
- `PlanningSteps.md` - Project planning and methodology documentation
- `PlanningPseaudoCode.md` - Detailed pseudocode for implementation

## Detailed File Index

### 01_Initial_Data/feature_engineering_visual.py
- Generates visualizations for feature engineering process
- Creates plots showing data distributions and relationships
- Outputs figures to the figures/ directory

### 02_Customer_Segments/02_Hierarchical_Clustering.py
- Implements hierarchical clustering algorithm
- Generates dendrograms and cluster visualizations
- Creates cluster summaries and exports results

### 02_Customer_Segments/create_cluster_summary.py
- Processes clustering results into summary statistics
- Generates cluster center heatmaps
- Exports cluster characteristics to CSV

### 03_KMF/02_KMF_COX.ipynb
- Implements Kaplan-Meier survival analysis
- Fits Cox Proportional Hazards model
- Generates survival curves and hazard ratios

### 05_RFM_Analysis/Telecom_RFM_segmentation.ipynb
- Performs RFM (Recency, Frequency, Monetary) analysis
- Creates customer segments based on RFM scores
- Generates visualizations of segment characteristics

## Usage Examples

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Install required packages
pip install -r requirements.txt
```

### Running Analysis
```python
# Example: Run customer segmentation
python 02_Customer_Segments/02_Hierarchical_Clustering.py

# Example: Generate RFM segments
jupyter notebook 05_RFM_Analysis/Telecom_RFM_segmentation.ipynb
```

## Contributing

We welcome contributions to improve this framework. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
