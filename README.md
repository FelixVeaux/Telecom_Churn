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

### 03_RFM_Analysis/
RFM (Recency, Frequency, Monetary) analysis:
- `Telecom_RFM_segmentation.ipynb` - RFM analysis and customer segmentation
- `rfm_variable_candidates.xlsx` - Potential RFM variables for analysis

### 04_KMF/
Kaplan-Meier and Cox Proportional Hazards analysis:
- `01_KMF_VariableSelection.ipynb` - Variable selection for survival analysis
- `02_KMF_COX.ipynb` - Implementation of KMF and Cox models
- `03_KMF_COX_CHURN_Prediction.ipynb` - Churn prediction using survival analysis
- `model_comparison.txt` - Comparison of different model performances
- `column_analysis_cleaneddf.csv` - Analysis of cleaned dataset columns
- `figures/` - Directory containing survival analysis visualizations

### 05_Chrun_Prediction/
Churn prediction models:
- `01_Chrun_Prediction.ipynb` - Implementation of churn prediction models

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

### 03_RFM_Analysis/Telecom_RFM_segmentation.ipynb
- Performs RFM (Recency, Frequency, Monetary) analysis
- Creates customer segments based on RFM scores
- Generates visualizations of segment characteristics

### 04_KMF/02_KMF_COX.ipynb
- Implements Kaplan-Meier survival analysis
- Fits Cox Proportional Hazards model
- Generates survival curves and hazard ratios

### 05_Chrun_Prediction/01_Chrun_Prediction.ipynb
- Generates churn predictions using survival analysis

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
jupyter notebook 03_RFM_Analysis/Telecom_RFM_segmentation.ipynb
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

## Data Flow

The project follows a sequential data processing pipeline:

1. **Data Ingestion & Cleaning** (`01_Initial_Data/`)
   - Raw data (`Cell1.csv`) → Cleaned data (`Churn_Data_cleaned.csv`)
   - Feature engineering and visualization generation
   - Output: Cleaned dataset and initial visualizations

2. **Customer Segmentation** (`02_Customer_Segments/`)
   - Input: Cleaned data from step 1
   - Process: Hierarchical clustering and analysis
   - Output: `Churn_Data_with_Clusters.csv` and cluster visualizations

3. **RFM Analysis** (`03_RFM_Analysis/`)
   - Input: Cleaned data from step 1
   - Process: Customer value assessment
   - Output: Customer value segments and insights

4. **Survival Analysis** (`04_KMF/`)
   - Input: Clustered data from step 2
   - Process: KMF and Cox model implementation
   - Output: Survival predictions and hazard ratios

5. **Churn Prediction** (`05_Chrun_Prediction/`)
   - Input: Results from previous steps
   - Process: Final model implementation
   - Output: Churn predictions and probabilities

## Key Dependencies

The project requires the following key Python packages:

### Core Dependencies
- `pandas` (>=1.3.0) - Data manipulation and analysis
- `numpy` (>=1.20.0) - Numerical computations
- `scikit-learn` (>=0.24.0) - Machine learning algorithms
- `lifelines` (>=0.26.0) - Survival analysis
- `matplotlib` (>=3.4.0) - Basic plotting
- `seaborn` (>=0.11.0) - Advanced visualization
- `jupyter` (>=1.0.0) - Interactive notebooks

### Optional Dependencies
- `plotly` - Interactive visualizations
- `statsmodels` - Statistical analysis
- `scipy` - Scientific computing

## Project Workflow

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```python
# Start with data cleaning
jupyter notebook 01_Initial_Data/01_Data_Cleaning.ipynb

# Generate feature visualizations
python 01_Initial_Data/feature_engineering_visual.py
```

### 3. Customer Segmentation
```python
# Run clustering analysis
jupyter notebook 02_Customer_Segments/01_Customer_Clustering.ipynb

# Generate cluster summaries
python 02_Customer_Segments/create_cluster_summary.py
```

### 4. RFM Analysis
```python
# Perform RFM analysis
jupyter notebook 03_RFM_Analysis/Telecom_RFM_segmentation.ipynb
```

### 5. Survival Analysis
```python
# Perform KMF and Cox analysis
jupyter notebook 04_KMF/02_KMF_COX.ipynb
```

### 6. Final Predictions
```python
# Generate churn predictions
jupyter notebook 05_Chrun_Prediction/01_Chrun_Prediction.ipynb
```

## Output Files

### Data Files
- `Churn_Data_cleaned.csv` - Cleaned dataset
- `Churn_Data_with_Clusters.csv` - Dataset with cluster assignments
- `cluster_summary.csv` - Cluster characteristics
- `column_analysis_cleaneddf.csv` - Column analysis results

### Visualization Files
- `feature_engineering_visual.png` - Feature engineering process
- `hierarchical_clustering_dendrogram.png` - Cluster hierarchy
- `cluster_centers_heatmap.png` - Cluster characteristics
- Various plots in `figures/` directories

### Model Outputs
- `model_comparison.txt` - Model performance comparison
- Survival curves in `04_KMF/figures/`
- Prediction results in `05_Chrun_Prediction/`

## Common Issues and Solutions

### 1. Memory Issues
- **Problem**: Large dataset processing
- **Solution**: 
  - Use `churn_data_clustered_sample.csv` for testing
  - Process data in chunks
  - Increase system swap space

### 2. Model Convergence
- **Problem**: Models not converging
- **Solution**:
  - Check data preprocessing
  - Adjust model parameters
  - Verify feature scaling

### 3. Visualization Issues
- **Problem**: Memory errors with large plots
- **Solution**:
  - Reduce figure size
  - Use sample data for testing
  - Implement plot batching

## Data Dictionary

### Key Variables

#### Customer Information
- `customer_id` - Unique identifier
- `tenure` - Months with service
- `contract_type` - Service contract type

#### Usage Metrics
- `monthly_charges` - Average monthly bill
- `total_charges` - Cumulative charges
- `data_usage` - Monthly data consumption

#### Service Features
- `internet_service` - Type of internet service
- `phone_service` - Phone service status
- `streaming_services` - Streaming service subscriptions

#### Target Variable
- `churn` - Binary indicator (1 = churned, 0 = retained)

## Visualization Guide

### 1. Feature Engineering Visualizations
- **Location**: `01_Initial_Data/figures/`
- **Purpose**: Understand data distributions
- **Key Insights**: Feature relationships and patterns

### 2. Cluster Visualizations
- **Location**: `02_Customer_Segments/diagrams/`
- **Purpose**: Customer segment analysis
- **Key Insights**: Segment characteristics and boundaries

### 3. RFM Analysis Charts
- **Location**: `03_RFM_Analysis/`
- **Purpose**: Customer value assessment
- **Key Insights**: High-value customer identification

### 4. Survival Analysis Plots
- **Location**: `04_KMF/figures/`
- **Purpose**: Churn probability over time
- **Key Insights**: Customer lifetime patterns

## Performance Metrics

### 1. Clustering Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Cluster Separation

### 2. Survival Analysis Metrics
- Concordance Index
- Log-likelihood
- AIC/BIC

### 3. Prediction Metrics
- Accuracy
- Precision/Recall
- ROC-AUC
- F1-Score

### 4. RFM Analysis Metrics
- Segment Stability
- Value Distribution
- Churn Rate by Segment
