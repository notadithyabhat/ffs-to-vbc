#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import dask.dataframe as dd  # For handling large datasets if needed
import numpy as np
import os
import matplotlib.pyplot as plt


# In[2]:


icd9_path = "all_icd9_codes.csv"
hcpcs_betos_path = "hcpcs_betos_2010.csv"

# Load ICD9 reference with latin1 encoding and error handling
df_icd9 = pd.read_csv(icd9_path, dtype=str, encoding="latin1", encoding_errors="replace")
print("ICD9 Reference loaded:", df_icd9.shape)
print(df_icd9.head())

# Load HCPCS/BETOS reference file similarly
df_hcpcs = pd.read_csv(hcpcs_betos_path, dtype=str, encoding="latin1", encoding_errors="replace")
print("HCPCS/BETOS Reference loaded:", df_hcpcs.shape)
print(df_hcpcs.head())


# In[3]:


path_cc_2008 = "2008_Chronic_Conditions_PUF.csv"   # CSV file for 2008
path_cc_2010 = "2010_Chronic_Conditions_PUF.xlsx"    # Excel file for 2010

# Load 2008 Chronic Conditions CSV
df_cc_2008 = pd.read_csv(path_cc_2008)
print("2008 Chronic Conditions:", df_cc_2008.shape)
print(df_cc_2008.head())

# Load 2010 Chronic Conditions Excel file correctly
xls = pd.ExcelFile(path_cc_2010)
df_cc_2010 = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
print("2010 Chronic Conditions shape:", df_cc_2010.shape)
print(df_cc_2010.head())


# In[4]:


ipbs_2008_path = "2008 IPBS PUF.csv"
ipbs_2010_path = "2010 IPBS PUF.csv"

# Specify PROVIDER_ID as string during loading for consistency
df_ipbs_2008 = pd.read_csv(ipbs_2008_path, dtype={"PROVIDER_ID": str})
print("IPBS 2008:", df_ipbs_2008.shape)
print(df_ipbs_2008.head())

df_ipbs_2010 = pd.read_csv(ipbs_2010_path, dtype={"PROVIDER_ID": str})
print("IPBS 2010:", df_ipbs_2010.shape)
print(df_ipbs_2010.head())


# In[5]:


psbs_path = "2010 PSBS PUF.csv"  # Path to PSBS CSV file

# Explicitly specify dtypes for all problematic columns:
dtype_spec = {
    "PROVIDER_ID": "str",
    "anemia_bene_cnt": "float64",
    "hyperl_bene_cnt": "float64",
    "hypert_bene_cnt": "float64",
    "ihd_bene_cnt": "float64",
    "nppes_entity_type_cd": "float64",
    "nppes_provider_mailing_zip": "object",
    "nppes_provider_practice_zip": "object",
    "provider_specialty": "object",
    "race_white_bene_cnt": "float64",
    "raoa_bene_cnt": "float64",
    "total_srvc_cnt": "float64"
}

# Read the file with the explicit dtype specifications.
df_psbs_dask = dd.read_csv(psbs_path, dtype=dtype_spec, blocksize="256MB")

print("PSBS (Dask) Schema:")
print(df_psbs_dask.dtypes)
print("\nApproximate Row Count:", df_psbs_dask.shape[0].compute())


# In[6]:


def clean_chronic_conditions(df):
    """
    Cleans a Chronic Conditions PUF DataFrame.
    - Strips whitespace from column names and converts them to uppercase.
    - Forces the AGE column to numeric if it exists (coercing invalid entries to NaN).
    - Fills missing numeric values with 0.
    - Fills missing object values with an empty string.
    """
    df.columns = [col.strip().upper() for col in df.columns]
    
    # Force 'AGE' to numeric if present
    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("")
    
    return df

def clean_ipbs(df):
    """
    Cleans an IPBS PUF DataFrame.
    - Converts column names to uppercase.
    - Converts count columns (e.g., those containing 'BENE_CNT') to numeric, fills NaNs with 0,
      and converts negative counts to 0.
    - Parses date columns (those with 'DT' in the name) to datetime.
    - Drops duplicates based on PROVIDER_ID.
    """
    df.columns = [col.upper() for col in df.columns]
    
    count_cols = [c for c in df.columns if "BENE_CNT" in c]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df[c] = df[c].apply(lambda x: x if x >= 0 else 0)
    
    date_cols = [col for col in df.columns if "DT" in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if "PROVIDER_ID" in df.columns:
        df.drop_duplicates(subset=["PROVIDER_ID"], inplace=True)
    
    return df

def clean_psbs(df):
    """
    Cleans a PSBS (Carrier) PUF DataFrame.
    - Standardizes column names.
    - Ensures zip code and provider specialty fields are strings.
    - Converts count fields to numeric and fixes negative values.
    - Drops duplicate provider records.
    
    Assumes df is a pandas DataFrame (e.g., computed from a Dask DataFrame).
    """
    df.columns = [col.upper() for col in df.columns]
    
    # Ensure PROVIDER_ID is a stripped string
    if "PROVIDER_ID" in df.columns:
        df["PROVIDER_ID"] = df["PROVIDER_ID"].astype(str).str.strip()
    
    # Ensure zip code columns are strings
    zip_cols = ["NPPES_PROVIDER_MAILING_ZIP", "NPPES_PROVIDER_PRACTICE_ZIP"]
    for col in zip_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Identify count columns (those containing 'BENE_CNT', 'EVENTS', or 'CNT')
    count_cols = [c for c in df.columns if ("BENE_CNT" in c) or ("EVENTS" in c) or ("CNT" in c)]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df[c] = df[c].apply(lambda x: x if x >= 0 else 0)
    
    if "PROVIDER_SPECIALTY" in df.columns:
        df["PROVIDER_SPECIALTY"] = df["PROVIDER_SPECIALTY"].astype(str)
    
    if "PROVIDER_ID" in df.columns:
        df.drop_duplicates(subset=["PROVIDER_ID"], inplace=True)
    
    return df

def clean_icd9(df):
    """
    Cleans an ICD9 codes DataFrame.
    - Standardizes column names.
    - Ensures all columns are treated as strings.
    """
    df.columns = [col.upper() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df

def clean_hcpcs(df):
    """
    Cleans an HCPCS/BETOS DataFrame.
    - Standardizes column names.
    - Ensures all columns are strings.
    """
    df.columns = [col.upper() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


# In[7]:


df_cc_2008_clean = clean_chronic_conditions(df_cc_2008)
print("Cleaned 2008 Chronic Conditions shape:", df_cc_2008_clean.shape)

df_cc_2010_clean = clean_chronic_conditions(df_cc_2010)
print("Cleaned 2010 Chronic Conditions shape:", df_cc_2010_clean.shape)

df_ipbs_2008_clean = clean_ipbs(df_ipbs_2008)
print("Cleaned 2008 IPBS shape:", df_ipbs_2008_clean.shape)

df_ipbs_2010_clean = clean_ipbs(df_ipbs_2010)
print("Cleaned 2010 IPBS shape:", df_ipbs_2010_clean.shape)

# For PSBS, sample a fraction from the Dask DataFrame and clean it
df_psbs_sample = df_psbs_dask.sample(frac=0.01, random_state=42).compute()
df_psbs_clean = clean_psbs(df_psbs_sample)
print("Cleaned PSBS sample shape:", df_psbs_clean.shape)

# Clean ICD9 and HCPCS/BETOS reference files
df_icd9_clean = clean_icd9(df_icd9)
print("Cleaned ICD9 Reference shape:", df_icd9_clean.shape)

df_hcpcs_clean = clean_hcpcs(df_hcpcs)
print("Cleaned HCPCS/BETOS Reference shape:", df_hcpcs_clean.shape)


# In[8]:


# Ensure the 'AGE' column exists
if "AGE" not in df_cc_2010_clean.columns:
    raise KeyError("Column 'AGE' not found in Chronic Conditions data. Available columns: " +
                   str(df_cc_2010_clean.columns.tolist()))

# Use pd.cut to create age categories
bins = [0, 64, 69, 74, 79, 84, np.inf]
labels = [1, 2, 3, 4, 5, 6]
df_cc_2010_clean["BENE_AGE_CAT_CD"] = pd.cut(df_cc_2010_clean["AGE"].astype(float), bins=bins, labels=labels, right=True)
df_cc_2010_clean = df_cc_2010_clean.dropna(subset=["BENE_AGE_CAT_CD"])
df_cc_2010_clean["BENE_AGE_CAT_CD"] = df_cc_2010_clean["BENE_AGE_CAT_CD"].astype(int)

# Aggregate Chronic Conditions data (2010) by age category
cc_metric_col = "COUNT OF BENEFICIARIES (PART A < 12)"
if cc_metric_col not in df_cc_2010_clean.columns:
    raise KeyError(f"Expected metric column '{cc_metric_col}' not found. Available columns: " +
                   str(df_cc_2010_clean.columns.tolist()))

cc_age_agg = (
    df_cc_2010_clean.groupby("BENE_AGE_CAT_CD", as_index=False)[cc_metric_col]
    .sum()
    .rename(columns={cc_metric_col: "CC_BENE_COUNT"})
)

print("Aggregated Chronic Conditions (by age group):")
print(cc_age_agg)


# In[9]:


# Aggregate IPBS 2010 data by HRR_PROV
if "HRR_PROV" not in df_ipbs_2010_clean.columns:
    raise KeyError("Column 'HRR_PROV' not found in IPBS data. Available columns: " +
                   str(df_ipbs_2010_clean.columns.tolist()))

ipbs_hrr_agg = df_ipbs_2010_clean.groupby("HRR_PROV").agg({
    "AGE_LESS_65_BENE_CNT": "sum",
    "AGE_65_69_BENE_CNT": "sum",
    "AGE_70_74_BENE_CNT": "sum",
    "AGE_75_79_BENE_CNT": "sum",
    "AGE_80_84_BENE_CNT": "sum",
    "AGE_OVER_84_BENE_CNT": "sum"
}).reset_index()

print("IPBS Aggregated by HRR_PROV:")
print(ipbs_hrr_agg.head())


# In[10]:


# Aggregate PSBS data by NPPES_PROVIDER_MAILING_ZIP.
if "NPPES_PROVIDER_MAILING_ZIP" not in df_psbs_clean.columns:
    raise KeyError("Column 'NPPES_PROVIDER_MAILING_ZIP' not found in PSBS data. Available columns: " +
                   str(df_psbs_clean.columns.tolist()))

psbs_zip_agg = df_psbs_clean.groupby("NPPES_PROVIDER_MAILING_ZIP").agg({
    "BENE_CNT": "sum",
    "TOTAL_SRVC_CNT": "sum",
    "TOTAL_EVENTS": "sum"  # Using TOTAL_EVENTS as a substitute for READMITS
}).reset_index()

print("PSBS Aggregated by NPPES_PROVIDER_MAILING_ZIP:")
print(psbs_zip_agg.head())


# In[11]:


# Option 1: Use TOTAL_EVENTS as a substitute for READMITS in aggregation
aggregation_columns_option1 = {
    "BENE_CNT": "sum",         
    "TOTAL_SRVC_CNT": "sum",     
    "TOTAL_EVENTS": "sum"  # Replacing READMITS with TOTAL_EVENTS
}

# Option 2: Exclude the READMITS column entirely
aggregation_columns_option2 = {
    "BENE_CNT": "sum",         
    "TOTAL_SRVC_CNT": "sum"
}

print("Using Option 1 (with TOTAL_EVENTS):")
try:
    df_psbs_agg_option1 = df_psbs_clean.groupby("PROVIDER_ID").agg(aggregation_columns_option1).reset_index()
    print(df_psbs_agg_option1.head())
except Exception as e:
    print("Error aggregating Option 1:", e)

print("\nUsing Option 2 (excluding READMITS):")
try:
    df_psbs_agg_option2 = df_psbs_clean.groupby("PROVIDER_ID").agg(aggregation_columns_option2).reset_index()
    print(df_psbs_agg_option2.head())
except Exception as e:
    print("Error aggregating Option 2:", e)


# In[12]:


# Plot beneficiary counts by HRR_PROV from IPBS
plt.figure(figsize=(12,6))
plt.bar(ipbs_hrr_agg["HRR_PROV"].astype(str).head(20), ipbs_hrr_agg["AGE_LESS_65_BENE_CNT"].head(20))
plt.xlabel("HRR_PROV (Sample)")
plt.ylabel("Total AGE_LESS_65_BENE_CNT")
plt.title("Sample IPBS Beneficiary Count by HRR_PROV")
plt.xticks(rotation=45)
plt.show()

# Plot beneficiary counts by Mailing Zip from PSBS
plt.figure(figsize=(12,6))
plt.bar(psbs_zip_agg["NPPES_PROVIDER_MAILING_ZIP"].astype(str).head(20), psbs_zip_agg["BENE_CNT"].head(20))
plt.xlabel("NPPES_PROVIDER_MAILING_ZIP (Sample)")
plt.ylabel("Total BENE_CNT")
plt.title("Sample PSBS Beneficiary Count by Mailing Zip")
plt.xticks(rotation=45)
plt.show()


# In[13]:


# At this point, you have:
# - cc_age_agg: Chronic Conditions data aggregated by age.
# - ipbs_hrr_agg: IPBS data aggregated by HRR_PROV.
# - psbs_zip_agg: PSBS data aggregated by mailing zip (using TOTAL_EVENTS as a proxy for READMITS).

# Print summary statistics for comparison:
print("Chronic Conditions Age Aggregation:")
print(cc_age_agg.describe())

print("\nIPBS Aggregation by HRR_PROV:")
print(ipbs_hrr_agg.describe())

print("\nPSBS Aggregation by Mailing Zip:")
print(psbs_zip_agg.describe())


# In[14]:


# Investigate the 'AGE' column in the Chronic Conditions data
print("Unique AGE values in df_cc_2010_clean:")
print(df_cc_2010_clean["AGE"].unique())

print("\nDescriptive statistics for AGE (after converting to numeric):")
age_numeric = pd.to_numeric(df_cc_2010_clean["AGE"], errors='coerce')
print(age_numeric.describe())

# Re-apply the age categorization
bins = [0, 64, 69, 74, 79, 84, np.inf]
labels = [1, 2, 3, 4, 5, 6]
df_cc_2010_clean["BENE_AGE_CAT_CD"] = pd.cut(age_numeric, bins=bins, labels=labels, right=True)

print("\nUnique Age Categories after categorization:")
print(df_cc_2010_clean["BENE_AGE_CAT_CD"].unique())

print("\nValue counts of Age Categories:")
print(df_cc_2010_clean["BENE_AGE_CAT_CD"].value_counts())


# In[15]:


cc_age_agg.head()


# In[16]:


print("df_cc_2010_clean shape:", df_cc_2010_clean.shape)
print("Columns in df_cc_2010_clean:", df_cc_2010_clean.columns.tolist())

# Ensure AGE is numeric
print("\nCheck AGE column dtype and some stats:")
print("AGE dtype:", df_cc_2010_clean["AGE"].dtype)
print(df_cc_2010_clean["AGE"].describe())

print("\nUnique AGE values (sample):")
print(df_cc_2010_clean["AGE"].unique()[:50])  # just show first 50 unique values if large

print("\nValue counts of AGE:")
print(df_cc_2010_clean["AGE"].value_counts())


# In[ ]:


# --- Flatten columns if the DataFrame came from Excel with a multi-level header ---
if isinstance(df_cc_2010_clean.columns, pd.MultiIndex):
    df_cc_2010_clean.columns = [
        "_".join(str(level) for level in col if str(level) != "")
        for col in df_cc_2010_clean.columns
    ]
    df_cc_2010_clean.reset_index(drop=True, inplace=True)

# --- Remove any existing "BENE_AGE_CAT_CD" to avoid duplication ---
if "BENE_AGE_CAT_CD" in df_cc_2010_clean.columns:
    df_cc_2010_clean.drop(columns=["BENE_AGE_CAT_CD"], inplace=True)

# --- Rename "AGE" -> "BENE_AGE_CAT_CD" ---
df_cc_2010_clean.rename(columns={"AGE": "BENE_AGE_CAT_CD"}, inplace=True)

# --- Ensure BENE_AGE_CAT_CD is numeric (if it isn't already) ---
df_cc_2010_clean["BENE_AGE_CAT_CD"] = (
    pd.to_numeric(df_cc_2010_clean["BENE_AGE_CAT_CD"], errors="coerce")
      .astype("Int64")
)

# --- Drop rows where the new column is NaN ---
df_cc_2010_clean.dropna(subset=["BENE_AGE_CAT_CD"], inplace=True)

# --- Verify the metric column exists ---
cc_metric_col = "COUNT OF BENEFICIARIES (PART A < 12)"
if cc_metric_col not in df_cc_2010_clean.columns:
    raise KeyError(
        f"Expected '{cc_metric_col}' not found. "
        f"Available columns: {df_cc_2010_clean.columns.tolist()}"
    )

# --- Group by the renamed age column and aggregate ---
cc_age_agg = (
    df_cc_2010_clean.groupby("BENE_AGE_CAT_CD", as_index=False)[cc_metric_col]
    .sum()
    .rename(columns={cc_metric_col: "CC_BENE_COUNT"})
)

print("Aggregated Chronic Conditions by Age Category:")
print(cc_age_agg)
print(cc_age_agg.describe())

