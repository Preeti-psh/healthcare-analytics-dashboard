#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[38]:


# Read CSV instead of Excel
df = pd.read_csv('For_python_notebook_Hospital_data.csv')
print(f"Loaded {len(df):,} records")


# In[9]:


# Step 2: Define mappings
measure_mapping = {
        'READM-30-AMI-HRRP': 'Acute Myocardial Infarction (Heart Attack)',
        'READM-30-HF-HRRP': 'Heart Failure', 
        'READM-30-PN-HRRP': 'Pneumonia',
        'READM-30-COPD-HRRP': 'COPD (Chronic Obstructive Pulmonary Disease)',
        'READM-30-THA-TKA-HRRP': 'Hip/Knee Replacement',
        'READM-30-CABG-HRRP': 'Coronary Artery Bypass Graft (CABG)'
    }
    
state_regions = {
        'AL': 'South', 'AK': 'West', 'AZ': 'West', 'AR': 'South', 'CA': 'West',
        'CO': 'West', 'CT': 'Northeast', 'DE': 'Northeast', 'FL': 'South', 'GA': 'South',
        'HI': 'West', 'ID': 'West', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest',
        'KS': 'Midwest', 'KY': 'South', 'LA': 'South', 'ME': 'Northeast', 'MD': 'Northeast',
        'MA': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest', 'MS': 'South', 'MO': 'Midwest',
        'MT': 'West', 'NE': 'Midwest', 'NV': 'West', 'NH': 'Northeast', 'NJ': 'Northeast',
        'NM': 'West', 'NY': 'Northeast', 'NC': 'South', 'ND': 'Midwest', 'OH': 'Midwest',
        'OK': 'South', 'OR': 'West', 'PA': 'Northeast', 'RI': 'Northeast', 'SC': 'South',
        'SD': 'Midwest', 'TN': 'South', 'TX': 'South', 'UT': 'West', 'VT': 'Northeast',
        'VA': 'South', 'WA': 'West', 'WV': 'South', 'WI': 'Midwest', 'WY': 'West',
        'DC': 'Northeast', 'PR': 'Other', 'VI': 'Other', 'GU': 'Other'
    }


# In[11]:


# Step 3: Clean the main dataset
print("Cleaning main dataset...")
df_clean = df.copy()


# In[12]:


# Clean and standardize facility names
df_clean['Facility_Name'] = df_clean['Facility Name'].str.strip()
df_clean['Facility_ID'] = df_clean['Facility ID'].astype(str)
df_clean['State'] = df_clean['State'].str.strip().str.upper()
df_clean['Region'] = df_clean['State'].map(state_regions).fillna('Other')


# In[14]:


# Create readable measure names
df_clean['Measure_Name'] = df_clean['Measure Name']
df_clean['Measure_Name_Clean'] = df_clean['Measure Name'].map(measure_mapping)
df_clean['Measure_Name_Clean'] = df_clean['Measure_Name_Clean'].fillna(df_clean['Measure Name'])


# In[15]:


# Create performance categories
def categorize_performance(err):
        if pd.isna(err):
            return 'No Data'
        elif err < 0.95:
            return 'Better than Expected'
        elif err <= 1.05:
            return 'As Expected'
        else:
            return 'Worse than Expected'

df_clean['Performance_Category'] = df_clean['Excess Readmission Ratio'].apply(categorize_performance)


# In[16]:


# Create volume categories
def categorize_volume(discharges):
        if pd.isna(discharges):
            return 'Unknown'
        elif discharges < 100:
            return 'Low Volume (<100)'
        elif discharges < 300:
            return 'Medium Volume (100-299)'
        else:
            return 'High Volume (300+)'

df_clean['Volume_Category'] = df_clean['Number of Discharges'].apply(categorize_volume)


# In[18]:


df_clean['Number of Readmissions'] = pd.to_numeric(df_clean['Number of Readmissions'], errors='coerce')
df_clean['Number of Discharges'] = pd.to_numeric(df_clean['Number of Discharges'], errors='coerce')


# In[19]:


# Calculate readmission rate safely (avoiding division by zero)
df_clean['Actual_Readmission_Rate_Percent'] = np.where(
    df_clean['Number of Discharges'] > 0,
    (df_clean['Number of Readmissions'] / df_clean['Number of Discharges'] * 100).round(2),
    np.nan
)


# In[20]:


#Clean numeric fields
df_clean['Number_of_Discharges'] = df_clean['Number of Discharges'].fillna(0)
df_clean['Number_of_Readmissions'] = df_clean['Number of Readmissions'].fillna(0)
df_clean['Excess_Readmission_Ratio'] = df_clean['Excess Readmission Ratio']
df_clean['Predicted_Readmission_Rate'] = df_clean['Predicted Readmission Rate']
df_clean['Expected_Readmission_Rate'] = df_clean['Expected Readmission Rate']


# In[21]:


# Clean date fields
df_clean['Start_Date'] = df_clean['Start Date']
df_clean['End_Date'] = df_clean['End Date']


# In[23]:


# Select final columns for main dataset
main_columns = [
        'Facility_Name', 'Facility_ID', 'State', 'Region', 'Measure_Name', 
        'Measure_Name_Clean', 'Number_of_Discharges', 'Excess_Readmission_Ratio',
        'Predicted_Readmission_Rate', 'Expected_Readmission_Rate', 
        'Number_of_Readmissions', 'Performance_Category', 'Volume_Category',
        'Actual_Readmission_Rate_Percent', 'Start_Date', 'End_Date'
    ]
    
main_dataset = df_clean[main_columns].copy()
    
print(f" Main dataset: {len(main_dataset):,} records")


# In[24]:


# Step 4: Create hospital summary
print("Creating hospital summary...")
hospital_summary = df_clean.groupby(['Facility_Name', 'Facility_ID', 'State', 'Region']).agg({
        'Excess_Readmission_Ratio': 'mean',
        'Number_of_Discharges': 'sum',
        'Number_of_Readmissions': 'sum',
        'Measure_Name': 'count'
    }).round(4)
    
hospital_summary.columns = ['Avg_ERR', 'Total_Discharges', 'Total_Readmissions', 'Measures_Count']
hospital_summary['Overall_Readmission_Rate'] = (
        hospital_summary['Total_Readmissions'] / hospital_summary['Total_Discharges'] * 100
    ).round(2)


# In[26]:


#Hospital size categories
def categorize_hospital_size(total_discharges):
        if total_discharges < 1000:
            return 'Small Hospital'
        elif total_discharges < 3000:
            return 'Medium Hospital'
        else:
            return 'Large Hospital'
    
hospital_summary['Hospital_Size_Category'] = hospital_summary['Total_Discharges'].apply(categorize_hospital_size)
hospital_summary = hospital_summary.reset_index()
    
print(f"Hospital summary: {len(hospital_summary):,} hospitals")


# In[39]:


# Step 5: Create state summary
print("Creating state summary...")
state_summary = df_clean.groupby(['State', 'Region']).agg({
        'Excess_Readmission_Ratio': 'mean',
        'Number_of_Discharges': 'sum',
        'Number_of_Readmissions': 'sum',
        'Facility_ID': 'nunique'
    }).round(4)
    
state_summary.columns = ['Avg_ERR', 'Total_Discharges', 'Total_Readmissions', 'Hospital_Count']
state_summary['State_Readmission_Rate'] = (
        state_summary['Total_Readmissions'] / state_summary['Total_Discharges'] * 100
    ).round(2)


# In[40]:


# Performance rating
def rate_performance(avg_err):
        if pd.isna(avg_err):
            return 'No Data'
        elif avg_err < 0.92:
            return 'Excellent'
        elif avg_err < 0.97:
            return 'Above Average'
        elif avg_err < 1.02:
            return 'Average'
        else:
            return 'Below Average'
    
state_summary['Performance_Rating'] = state_summary['Avg_ERR'].apply(rate_performance)
state_summary = state_summary.reset_index()
    
print(f" State summary: {len(state_summary):,} states")


# In[29]:


# Step 6: Create measure summary
print("Creating measure summary...")
measure_summary = df_clean.groupby(['Measure_Name', 'Measure_Name_Clean']).agg({
        'Excess_Readmission_Ratio': ['mean', 'std', 'count'],
        'Number_of_Discharges': 'sum',
        'Number_of_Readmissions': 'sum'
    }).round(4)
    
measure_summary.columns = ['Avg_ERR', 'Std_ERR', 'Hospital_Count', 'Total_Discharges', 'Total_Readmissions']
measure_summary['National_Readmission_Rate'] = (
        measure_summary['Total_Readmissions'] / measure_summary['Total_Discharges'] * 100
    ).round(2)


# In[41]:


# Risk level
def assign_risk_level(readmission_rate):
        if readmission_rate < 12:
            return 'Low'
        elif readmission_rate < 17:
            return 'Medium'
        else:
            return 'High'
    
measure_summary['Risk_Level'] = measure_summary['National_Readmission_Rate'].apply(assign_risk_level)
measure_summary['CMS_Target_ERR'] = 1.00  # National benchmark
measure_summary = measure_summary.reset_index()
    
print(f" Measure summary: {len(measure_summary):,} conditions")


# In[42]:


# Step 7: Save all files
print("\n💾 Saving CSV files...")
    
files_created = []
    
# Save main dataset
main_file = "cms_readmissions_main.csv"
main_dataset.to_csv(main_file, index=False)
files_created.append(main_file)
print(f"{main_file} ({len(main_dataset):,} rows)")
    
# Save hospital summary
hospital_file = "cms_readmissions_hospital_summary.csv"
hospital_summary.to_csv(hospital_file, index=False)
files_created.append(hospital_file)
print(f"{hospital_file} ({len(hospital_summary):,} rows)")
    
# Save state summary
state_file = "cms_readmissions_state_summary.csv"
state_summary.to_csv(state_file, index=False)
files_created.append(state_file)
print(f"{state_file} ({len(state_summary):,} rows)")
    
# Save measure summary
measure_file = "cms_readmissions_measure_summary.csv"
measure_summary.to_csv(measure_file, index=False)
files_created.append(measure_file)
print(f"{measure_file} ({len(measure_summary):,} rows)")


# In[44]:


# Step 8: Summary statistics
print("\n DATA SUMMARY:")
print("=" * 40)
print(f"Total hospitals: {main_dataset['Facility_ID'].nunique():,}")
print(f" States covered: {main_dataset['State'].nunique()}")
print(f" Medical conditions: {main_dataset['Measure_Name_Clean'].nunique()}")
print(f" Total records: {len(main_dataset):,}")

print("\n Performance Distribution:")
perf_dist = main_dataset['Performance_Category'].value_counts()
for category, count in perf_dist.items():
    percentage = count / len(main_dataset) * 100
    print(f"   {category}: {count:,} ({percentage:.1f}%)")

print("\n Regional Distribution:")
region_dist = main_dataset['Region'].value_counts()
for region, count in region_dist.items():
    print(f"   {region}: {count:,} records")

print(f"\n ALL DONE! Created {len(files_created)} CSV files ready for Tableau")
print("\nFiles created:")
for file in files_created:
    print(f"    {file}")

# Instead of return, just print completion message
print("\n Data cleaning and analysis complete!")


# In[ ]:




