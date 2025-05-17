# Telecom Churn Analysis & Budget Allocation

## 1. Introduction

Customer churn represents a significant challenge in the telecom industry, where acquiring new customers costs 5-25 times more than retaining existing ones. This repository provides a comprehensive workflow for telecom companies to analyze customer churn patterns, predict at-risk customers, and strategically allocate promotion budgets to maximize retention.

This framework serves:
- **Data Scientists**: For implementing predictive models and survival analysis
- **Business Analysts**: For customer segmentation and insights generation
- **Marketing Teams**: For targeted promotion planning and budget optimization

## 2. Data Preparation

### Step 1: Data Collection

Typical data sources include CRM systems, billing platforms, call detail records, network logs, and customer support tickets.

Raw data often contains:
```
customer_id, join_date, contract_type, monthly_charges, call_minutes, data_usage, 
support_calls, demographics, payment_history, service_outages
```

### Step 2: Data Cleaning

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('telecom_data.csv')

# Handle missing values
df['data_usage'] = df['data_usage'].fillna(df['data_usage'].median())

# Remove duplicates
df = df.drop_duplicates(subset=['customer_id'])

# Fix data types
df['join_date'] = pd.to_datetime(df['join_date'])

# Calculate tenure in months
df['tenure_months'] = ((pd.Timestamp.now() - df['join_date']).dt.days / 30).astype(int)

# Create churn label (example)
df['churned'] = np.where(df['account_status'] == 'Closed', 1, 0)
```

### Step 3: Feature Engineering

```python
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Normalize numerical features
scaler = MinMaxScaler()
numerical_cols = ['monthly_charges', 'call_minutes', 'data_usage', 'tenure_months']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# One-hot encode categorical variables
categorical_cols = ['contract_type', 'payment_method', 'service_plan']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df, encoded_df], axis=1)

# Create derived features
df['usage_change_rate'] = df['data_usage'] / df['data_usage_6m_ago']
df['complaint_rate'] = df['support_calls'] / df['tenure_months']
df['monthly_revenue'] = df['monthly_charges'] * (1 - df['discount_rate'])
```

## 3. Customer Segmentation

### Step 4: Segment Customers

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Select segmentation variables
segment_features = ['tenure_months', 'monthly_revenue', 'data_usage', 
                    'complaint_rate', 'contract_type_Annual']

# Standardize data for clustering
X = df[segment_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters (elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')

# Apply K-Means with optimal clusters (k=4 for example)
kmeans = KMeans(n_clusters=4, random_state=42)
df['segment'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='monthly_revenue', y='tenure_months', 
                hue='segment', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments by Revenue and Tenure')
plt.savefig('customer_segments.png')

# Analyze segment characteristics
segment_profile = df.groupby('segment').agg({
    'monthly_revenue': 'mean',
    'tenure_months': 'mean',
    'data_usage': 'mean',
    'complaint_rate': 'mean',
    'churned': 'mean'
}).reset_index()
```

## 4. Survival Analysis

### Step 5: Kaplan-Meier Estimator

```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Create duration and event variables
df['duration'] = df['tenure_months']
df['event'] = df['churned']  # 1 if churned, 0 if still active

# Fit KM model
kmf = KaplanMeierFitter()
kmf.fit(df['duration'], df['event'], label='Overall')

# Plot overall survival curve
plt.figure(figsize=(12, 8))
kmf.plot_survival_function()
plt.title('Customer Survival Probability Over Time')
plt.xlabel('Months')
plt.ylabel('Probability of Remaining a Customer')

# Compare survival curves by segment
plt.figure(figsize=(12, 8))
for segment in df['segment'].unique():
    mask = df['segment'] == segment
    kmf.fit(df.loc[mask, 'duration'], df.loc[mask, 'event'], label=f'Segment {segment}')
    kmf.plot_survival_function()

plt.title('Survival Curves by Customer Segment')
plt.xlabel('Months')
plt.ylabel('Probability of Remaining a Customer')
plt.savefig('survival_curves.png')
```

### Step 6: Cox Proportional Hazards Model

```python
from lifelines import CoxPHFitter

# Prepare data for Cox PH model
cph_columns = ['duration', 'event', 'monthly_revenue', 'data_usage', 
               'complaint_rate', 'contract_type_Annual']
cph_data = df[cph_columns].copy()

# Fit Cox PH model
cph = CoxPHFitter()
cph.fit(cph_data, duration_col='duration', event_col='event')

# Print summary
cph.print_summary()

# Plot hazard ratios
plt.figure(figsize=(12, 8))
cph.plot()
plt.title('Hazard Ratios with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('hazard_ratios.png')

# Test proportional hazards assumption
cph.check_assumptions(cph_data, show_plots=True)
```

## 5. Churn Prediction

### Step 7: Build a Churn Prediction Model

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare features and target
feature_cols = ['tenure_months', 'monthly_revenue', 'data_usage', 'complaint_rate',
                'contract_type_Annual', 'contract_type_Monthly', 'payment_method_Credit']
X = df[feature_cols]
y = df['churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance for Churn Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Add prediction probabilities to dataset
df['churn_probability'] = rf_model.predict_proba(df[feature_cols])[:, 1]

# XGBoost model (alternative approach)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4)
xgb_model.fit(X_train, y_train)
print(f"XGBoost AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]):.4f}")
```

## 6. Budget Allocation & Promotion Planning

### Step 8: Identify High-Risk Segments

```python
# Calculate lifetime value (simplified example)
avg_monthly_revenue = df['monthly_revenue'].mean()
avg_customer_lifespan = 1 / df['churned'].mean()  # Inverse of churn rate
df['ltv'] = avg_monthly_revenue * avg_customer_lifespan

# Identify high-risk high-value customers
df['risk_score'] = df['churn_probability'] * df['ltv']
high_risk = df[df['churn_probability'] > 0.7]
high_value = df[df['ltv'] > df['ltv'].quantile(0.75)]
priority_customers = df[(df['churn_probability'] > 0.7) & (df['ltv'] > df['ltv'].quantile(0.75))]

print(f"Number of priority customers: {len(priority_customers)}")
```

### Step 9: Allocate Promotion Budget

```python
# Define total promotion budget
total_budget = 100000  # Example budget

# Calculate segment weights based on risk and value
df['segment_weight'] = df['churn_probability'] * (df['ltv'] / df['ltv'].max())

# Normalize weights to sum to 1
total_weight = df['segment_weight'].sum()
df['budget_allocation'] = (df['segment_weight'] / total_weight) * total_budget

# Group by segment for strategic planning
segment_budget = df.groupby('segment').agg({
    'budget_allocation': 'sum',
    'churn_probability': 'mean',
    'ltv': 'mean',
    'customer_id': 'count'
}).rename(columns={'customer_id': 'customer_count'})

print(segment_budget)

# Calculate ROI estimate
promotion_effectiveness = 0.3  # Assume 30% effectiveness in preventing churn
df['expected_return'] = df['budget_allocation'] * promotion_effectiveness * df['ltv']
df['estimated_roi'] = df['expected_return'] / df['budget_allocation']
```

### Step 10: Design Tailored Campaigns

```python
# Categorize customers for different promotion types
df['promotion_type'] = 'Standard'

# Data-focused customers (high data usage)
mask = (df['data_usage'] > df['data_usage'].quantile(0.75))
df.loc[mask, 'promotion_type'] = 'Data Booster'

# Price-sensitive customers (high churn probability, lower revenue)
mask = (df['churn_probability'] > 0.6) & (df['monthly_revenue'] < df['monthly_revenue'].median())
df.loc[mask, 'promotion_type'] = 'Price Discount'

# Service-sensitive customers (high complaint rate)
mask = df['complaint_rate'] > df['complaint_rate'].quantile(0.75)
df.loc[mask, 'promotion_type'] = 'Service Upgrade'

# Loyalty customers (high tenure but rising churn risk)
mask = (df['tenure_months'] > 24) & (df['churn_probability'] > 0.5)
df.loc[mask, 'promotion_type'] = 'Loyalty Rewards'

# Budget allocation by promotion type
promotion_budget = df.groupby('promotion_type').agg({
    'budget_allocation': 'sum',
    'customer_id': 'count'
}).rename(columns={'customer_id': 'customer_count'})

print(promotion_budget)
```

### Step 11: Conduct A/B Tests

```python
import numpy as np

# Create control and treatment groups
np.random.seed(42)
df['test_group'] = np.random.choice(['Control', 'Treatment'], size=len(df), p=[0.5, 0.5])

# Only apply promotions to treatment group
df['apply_promotion'] = (df['test_group'] == 'Treatment') & (df['churn_probability'] > 0.5)

# Set up tracking for conversion metrics (to be populated after campaign)
df['remained_customer'] = np.nan
df['days_extended'] = np.nan

# Export customer lists for campaign implementation
treatment_export = df[df['apply_promotion']].sort_values('churn_probability', ascending=False)
treatment_export[['customer_id', 'promotion_type', 'budget_allocation']].to_csv('promotion_targets.csv', index=False)
```

## 7. Monitoring and Feedback Loop

### Step 12: Track Key Metrics

```python
# Assuming data collection after campaign period
# This would be run after some time has passed

# Example code to calculate retention uplift
def calculate_metrics(df, post_campaign_data):
    # Merge post-campaign data
    df_merged = df.merge(post_campaign_data, on='customer_id', how='left')
    
    # Calculate retention rates
    control_retention = df_merged[df_merged['test_group'] == 'Control']['still_active'].mean()
    treatment_retention = df_merged[df_merged['test_group'] == 'Treatment']['still_active'].mean()
    
    # Calculate uplift
    retention_uplift = treatment_retention - control_retention
    
    # Calculate efficiency
    cost_per_retained = df_merged[df_merged['apply_promotion']]['budget_allocation'].sum() / \
                        (df_merged[(df_merged['apply_promotion']) & 
                                 (df_merged['still_active'] == 1)].shape[0])
    
    # Return key metrics
    return {
        'control_retention': control_retention,
        'treatment_retention': treatment_retention,
        'retention_uplift': retention_uplift,
        'cost_per_retained': cost_per_retained,
        'roi': (retention_uplift * df['ltv'].mean()) / cost_per_retained
    }

# Example dashboard visualization code
def plot_metrics_dashboard(metrics_over_time):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot retention rate over time
    axes[0, 0].plot(metrics_over_time['date'], metrics_over_time['treatment_retention'], 
                   label='Treatment', marker='o')
    axes[0, 0].plot(metrics_over_time['date'], metrics_over_time['control_retention'], 
                   label='Control', marker='o')
    axes[0, 0].set_title('Retention Rate Over Time')
    axes[0, 0].legend()
    
    # Plot uplift over time
    axes[0, 1].plot(metrics_over_time['date'], metrics_over_time['retention_uplift'], marker='o')
    axes[0, 1].set_title('Retention Uplift Over Time')
    
    # Plot cost efficiency over time
    axes[1, 0].plot(metrics_over_time['date'], metrics_over_time['cost_per_retained'], marker='o')
    axes[1, 0].set_title('Cost per Retained Customer')
    
    # Plot ROI over time
    axes[1, 1].plot(metrics_over_time['date'], metrics_over_time['roi'], marker='o')
    axes[1, 1].set_title('Return on Investment')
    
    plt.tight_layout()
    plt.savefig('metrics_dashboard.png')
```

### Step 13: Adjust Strategy

```python
# Example code for strategy adjustment pipeline
def update_model_and_strategy(df, new_data, current_model):
    """
    Update prediction models and promotion strategy based on new data
    """
    # Combine historical and new data
    updated_df = pd.concat([df, new_data], ignore_index=True)
    
    # Refresh features
    updated_df = engineer_features(updated_df)  # Feature engineering function
    
    # Retrain model
    features, target = prepare_model_data(updated_df)
    new_model = retrain_model(features, target, current_model)
    
    # Update customer segments
    updated_df = update_segmentation(updated_df)
    
    # Generate new promotion recommendations
    new_strategy = optimize_budget_allocation(updated_df, new_model)
    
    return updated_df, new_model, new_strategy

# Example of automated workflow with Airflow (pseudocode)
"""
# DAG definition
with DAG('telecom_churn_workflow', 
         schedule_interval='@monthly',
         default_args=default_args) as dag:
    
    # Extract new data
    extract_task = PythonOperator(
        task_id='extract_new_data',
        python_callable=extract_data
    )
    
    # Update models
    update_model_task = PythonOperator(
        task_id='update_model',
        python_callable=update_model_and_strategy
    )
    
    # Generate new campaign lists
    generate_campaign_task = PythonOperator(
        task_id='generate_campaign',
        python_callable=generate_campaign_lists
    )
    
    # Generate reports
    reporting_task = PythonOperator(
        task_id='generate_reports',
        python_callable=generate_performance_reports
    )
    
    # Set task dependencies
    extract_task >> update_model_task >> generate_campaign_task >> reporting_task
"""
```

## 8. Appendix & Notes

### Glossary

- **Churn**: When a customer terminates their service with the company
- **ARPU**: Average Revenue Per User
- **LTV**: Lifetime Value - estimated total revenue from a customer
- **Survival analysis**: Statistical approach to analyze the expected time until churn
- **Hazard ratio**: Relative likelihood of churn for different factors
- **Uplift**: Improvement in retention due to promotions
- **A/B testing**: Experimental approach to compare control and treatment groups

### Data Schema Example

```
Table: customers
- customer_id (PK)
- join_date
- demographics (age, gender, location)
- contract_type
- plan_details
- status (active/churned)
- churn_date (if applicable)

Table: usage
- usage_id (PK)
- customer_id (FK)
- month
- call_minutes
- sms_count
- data_gb
- overage_charges

Table: payments
- payment_id (PK)
- customer_id (FK)
- payment_date
- amount
- payment_method
- status (successful/failed)

Table: support
- ticket_id (PK)
- customer_id (FK)
- date
- issue_type
- resolution_time
- satisfaction_score
```

### Key Libraries

- Data manipulation: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- Machine learning: [scikit-learn](https://scikit-learn.org/), [xgboost](https://xgboost.readthedocs.io/)
- Survival analysis: [lifelines](https://lifelines.readthedocs.io/)
- Visualization: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
- Workflow automation: [Apache Airflow](https://airflow.apache.org/)

---

## Getting Started

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Place your telecom data in the `data/` directory
4. Run the complete workflow: `python src/main.py`
5. View results in the `output/` directory

For a more detailed walkthrough, see the Jupyter notebooks in the `notebooks/` directory. 