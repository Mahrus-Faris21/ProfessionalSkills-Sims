#!/usr/bin/env python
# coding: utf-8

# # Here is the background information on your task

# Sprocket Central Pty Ltd , a medium size bikes & cycling accessories organisation, has approached Tony Smith (Partner) in KPMG’s Lighthouse & Innovation Team. Sprocket Central Pty Ltd  is keen to learn more about KPMG’s expertise in its Analytics, Information & Modelling team. Smith discusses KPMG’s expertise in this space (you can read more here). In particular, he speaks about how the team can effectively analyse the datasets to help Sprocket Central Pty Ltd grow its business.
# 
# Primarily, Sprocket Central Pty Ltd needs help with its customer and transactions data. The organisation has a large dataset relating to its customers, but their team is unsure how to effectively analyse it to help optimise its marketing strategy. However, in order to support the analysis, you speak to the Associate Director for some ideas and she advised that “the importance of optimising the quality of customer datasets cannot be underestimated. The better the quality of the dataset, the better chance you will be able to use it drive company growth.”
# 
# The client provided KPMG with 3 datasets:
# - Customer Demographic 
# - Customer Addresses
# - Transactions data in the past 3 months
# 
# You decide to start the preliminary data exploration and identify ways to improve the quality of Sprocket Central Pty Ltd’s data.

# # Here is your task

# **You arrive at your desk after the initial client meeting. You have a voicemail on your phone which contains the following instructions.**
# 
# **[Voicemail transcript below]**
#  
# 
# “Hi there – Welcome again to the team! The client has asked our team to assess the quality of their data; as well as make recommendations on ways to clean the underlying data and mitigate these issues.  Can you please take a look at the datasets we’ve received and draft an email to them identifying the data quality issues and how this may impact our analysis going forward?
# 
# I will send through an example of a typical data quality framework that can be used as a guide. Remember to consider the join keys between the tables too. Thanks again for your help.”
# 
# 
# **[Read email below]**
# 
# "Hi there,
# 
# As per voicemail, please find the 3 datasets attached from Sprocket Central Pty Ltd:
# - Customer Demographic 
# - Customer Addresses
# - Transaction data in the past three months
# Can you please review the data quality to ensure that it is ready for our analysis in phase two. Remember to take note of any assumptions or issues we need to go back to the client on. As well as recommendations going forward to mitigate current data quality concerns.
# 
# I’ve also attached a data quality framework as a guideline. Let me know if you have any questions.
# 
# Thanks for your help."
# 
# Kind Regards,
# 
# Your Manager
# 
# **Here is your task:**
# 
# Draft an email to the client identifying the data quality issues and strategies to mitigate these issues. Refer to ‘Data Quality Framework Table’ and resources below for criteria and dimensions which you should consider.

# # Here are some resources to help you

# **Data Quality Framework Table**
# 
# Below is a list of the Data Quality dimensions our team may use to evaluate a dataset. Some of these terms are common to the whole industry, so you may find more information and clarity on these terms by searching online.

# <img src="https://cdn-assets.theforage.com/vinternship_modules/kpmg_data_analytics/Screen+Shot+2018-03-20+at+2.50.59+pm.png" alt="Alt Text" width="400" height="550">

# In[87]:


import pandas as pd


# In[148]:


# Specify the sheet Customer Demographic
sheet_name_to_read = 'CustomerDemographic'

# Provide the full file path
file_path = r'C:\Users\Sweet\Downloads\Portofolio\KPMG_VI_New_raw_data_update_final.xlsx'
SprocketCentral = pd.read_excel(file_path, sheet_name=sheet_name_to_read)
new_header = SprocketCentral.iloc[0]
SprocketCentral = SprocketCentral[1:]
SprocketCentral.columns = new_header

# Reset the index
SprocketCentral.reset_index(drop=True, inplace=True)
SprocketCentral.head()


# # Counting the Number of Unidentified Column Values

# In[164]:


SprocketCentral['first_name'].isna().sum()


# In[3]:


SprocketCentral['last_name'].isna().sum()


# In[15]:


SprocketCentral['job_title'].isna().sum()


# In[16]:


SprocketCentral['job_industry_category'].isna().sum()


# In[29]:


SprocketCentral['DOB'].isna().sum()


# In[40]:


SprocketCentral['tenure'].isna().sum()


# In[25]:


unique_genders = (SprocketCentral['gender'] == 'F').sum()
unique_genders


# In[30]:


unique_genders = (SprocketCentral['gender'] == 'U').sum()
unique_genders


# # Verifying Anomaly Data

# In[149]:


SprocketCentral['DOB'] = pd.to_datetime(SprocketCentral['DOB'])
SprocketCentral.sort_values(by='DOB', ascending=True)


# In[150]:


SprocketCentral.sort_index().head()


# In[153]:


# Drop rows where the specified column is empty or contains no content
columns_to_check = ['last_name', 'job_title', 'job_industry_category', 'wealth_segment', 'tenure']
SprocketCentral = SprocketCentral.dropna(subset=columns_to_check, how='all')

values_to_drop = ['F', 'U', 'NaN']
SprocketCentral = SprocketCentral[~SprocketCentral['gender'].isin(values_to_drop)]
SprocketCentral = SprocketCentral[~SprocketCentral['job_title'].isin(values_to_drop)]
SprocketCentral = SprocketCentral[~SprocketCentral['job_industry_category'].isin(values_to_drop)]

def drop_rows_with_nan_or_empty(df, column_name):
    df = df.dropna(subset=[column_name], how='any')
    return df

drop_rows_with_nan_or_empty(SprocketCentral, 'last_name')
drop_rows_with_nan_or_empty(SprocketCentral, 'job_title')
drop_rows_with_nan_or_empty(SprocketCentral, 'job_industry_category')
drop_rows_with_nan_or_empty(SprocketCentral, 'tenure')

del SprocketCentral['default']
SprocketCentral


# In[76]:


def get_unique_elements_and_counts(df, column_name):
    # Get unique elements in the specified column
    unique_elements = df[column_name].unique()
    
    # Get unique value counts in the specified column
    value_counts = df[column_name].value_counts()
    
    return unique_elements, value_counts


# In[83]:


def count_value_in_column(df, column_name, target_value):
    value_counts = df[column_name].value_counts()
    if target_value in value_counts.index:
        count = value_counts[target_value]
    else:
        count = 0
    
    return count


# In[154]:


get_unique_elements_and_counts(SprocketCentral, 'gender')


# In[161]:


count_value_in_column(SprocketCentral, 'last_name', '')


# In[156]:


get_unique_elements_and_counts(SprocketCentral, 'past_3_years_bike_related_purchases')


# In[157]:


get_unique_elements_and_counts(SprocketCentral, 'job_title')


# In[158]:


get_unique_elements_and_counts(SprocketCentral, 'job_industry_category')


# In[159]:


get_unique_elements_and_counts(SprocketCentral, 'wealth_segment')


# In[162]:


get_unique_elements_and_counts(SprocketCentral, 'owns_car')


# In[163]:


smallest_year = SprocketCentral['DOB'].dt.year.min()
largest_year = SprocketCentral['DOB'].dt.year.max()
# Print the results
print(f"Smallest Year: {smallest_year}")
print(f"Largest Year: {largest_year}")


# # Perform Standard Data Quality Dimensions

# In[37]:


#Accuracy (Correct Values)
def check_accuracy(df):
    # In this example, we check if Age is within a valid range (e.g., 1953-2002)
    df['DOB'] = pd.to_datetime(df['DOB'])
    valid_age_range = (df['DOB'].dt.year >= 1953) & (df['DOB'].dt.year <= 2002)
    return valid_age_range.all()

#Completeness (Data Fields with Values)
def check_completeness(df):
    # Check if there are any missing values in the DataFrame
    return not df.isnull().any().any()

#Consistency (Values Free from Contradiction)
def check_consistency(df):
    # Convert 'tenure' column to numeric (assuming it contains numeric values)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    
    # Define consistency conditions (e.g., tenure should be greater than 1)
    valid_tenure = df['tenure'] > 1
    return valid_tenure.all()


# In[27]:


check_customerdemographics = check_accuracy(SprocketCentral)
check_customerdemographics


# In[24]:


completeness_customerdemographics = check_completeness(SprocketCentral)
completeness_customerdemographics


# In[36]:


consistency_customerdemographics = check_consistency(SprocketCentral)
consistency_customerdemographics


# In[38]:


# Specify the sheet Customer Address
sheet_name_to_read = 'CustomerAddress'

# Provide the full file path
file_path = r'C:\Users\Sweet\Downloads\Portofolio\KPMG_VI_New_raw_data_update_final.xlsx'
SprocketCentral2 = pd.read_excel(file_path, sheet_name=sheet_name_to_read)
new_header = SprocketCentral2.iloc[0]
SprocketCentral2 = SprocketCentral2[1:]
SprocketCentral2.columns = new_header

# Reset the index
SprocketCentral2.reset_index(drop=True, inplace=True)
SprocketCentral2.head()


# In[39]:


SprocketCentral2['address'].isna().sum()


# In[42]:


SprocketCentral2['postcode'].isna().sum()


# In[41]:


SprocketCentral2['state'].isna().sum()


# In[80]:


get_unique_elements_and_counts(SprocketCentral2, 'state')


# In[46]:


get_unique_elements_and_counts(SprocketCentral2, 'address')


# In[47]:


get_unique_elements_and_counts(SprocketCentral2, 'property_valuation')


# In[50]:


count_value_in_column(SprocketCentral2, 'address', '8194 Lien Street')


# In[67]:


# Specify the sheet Customer Address
sheet_name_to_read = 'Transactions'

# Provide the full file path
file_path = r'C:\Users\Sweet\Downloads\Portofolio\KPMG_VI_New_raw_data_update_final.xlsx'
SprocketCentral3 = pd.read_excel(file_path, sheet_name=sheet_name_to_read)
new_header = SprocketCentral3.iloc[0]
SprocketCentral3 = SprocketCentral3[1:]
SprocketCentral3.columns = new_header
SprocketCentral3['transaction_date'] = pd.to_datetime(SprocketCentral3['transaction_date'])
# Reset the index
SprocketCentral3.reset_index(drop=True, inplace=True)
SprocketCentral3


# In[53]:


get_unique_elements_and_counts(SprocketCentral3, 'online_order')


# In[54]:


get_unique_elements_and_counts(SprocketCentral3, 'order_status')


# In[55]:


get_unique_elements_and_counts(SprocketCentral3, 'brand')


# In[56]:


get_unique_elements_and_counts(SprocketCentral3, 'product_line')


# In[57]:


get_unique_elements_and_counts(SprocketCentral3, 'product_class')


# In[63]:


count_value_in_column(SprocketCentral3, 'product_class', 'nan')


# In[68]:


get_unique_elements_and_counts(SprocketCentral3, 'product_class')


# In[69]:


get_unique_elements_and_counts(SprocketCentral3, 'product_size')


# In[70]:


get_unique_elements_and_counts(SprocketCentral3, 'list_price')


# In[71]:


get_unique_elements_and_counts(SprocketCentral3, 'standard_cost')


# In[72]:


get_unique_elements_and_counts(SprocketCentral3, 'product_first_sold_date')

