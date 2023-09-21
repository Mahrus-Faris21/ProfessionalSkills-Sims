#!/usr/bin/env python
# coding: utf-8

# # Here is the background information on your task

# The BCG project team thinks that building a churn model to understand whether price sensitivity is the largest driver of churn has potential. The client has sent over some data and the AD wants you to perform some exploratory data analysis.
# The data that was sent over includes: 
# - Historical customer data: Customer data such as usage, sign up date, forecasted usage etc
# - Historical pricing data: variable and fixed pricing data etc
# - Churn indicator: whether each customer has churned or not
# 
# Please submit analysis in a code script, notebook, or PDF format. 
# Please note, there are multiple ways to approach the task and that the sample answer is just one way to do it.

# # Here is your task

# **Sub-Task 1:**
# Perform some exploratory data analysis. Look into the data types, data statistics, specific parameters, and variable distributions. This first subtask is for you to gain a holistic understanding of the dataset. You should spend around 1 hour on this.

# **Sub-Task 2:**
# Verify the hypothesis of price sensitivity being to some extent correlated with churn. It is up to you to define price sensitivity and calculate it. You should spend around 30 minutes on this.
 
# **Sub-Task 3:**
# Prepare a half-page summary or slide of key findings and add some suggestions for data augmentation – which other sources of data should the client provide you with and which open source datasets might be useful? You should spend 10-15 minutes on this.
 
# For your final deliverable, please submit your analysis (in the form of a jupyter notebook, code script or PDF) as well as your half-page summary document.
 
# **Note:** Use the 2 datasets within the additional resources for this task and if you’re unsure on where to start with visualizing data, use the accompanying links. Be sure to also use the data description document to understand what the columns represent. The task description document outlines the higher-level motivation of the project. Finally, use the eda_starter.ipynb file to get started with some helper functions and methods.

# **If you are stuck:** Think about ways you can define price sensitivity. Make sure to think of all possible ways and investigate them.

# # The Answers

# In[28]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[12]:
file_path = r'C:\Users\Sweet\Downloads\Portofolio\client_data.csv'

# Read the CSV file into a DataFrame
client_data = pd.read_csv(file_path)

# Reset the index and drop the old index
client_data.reset_index(drop=True, inplace=True)

# Display the first few rows of the DataFrame
client_data.head()


# In[2]:
file_path = r'C:\Users\Sweet\Downloads\Portofolio\price_data.csv'

# Read the CSV file into a DataFrame
price_data = pd.read_csv(file_path)

# Reset the index and drop the old index
price_data.reset_index(drop=True, inplace=True)

# Display the first few rows of the DataFrame
price_data.head()


# In[14]:
price_data.shape


# **SUB-TASK 1:**

# In[3]:
price_data.info()

# In[6]:
print(price_data.isnull().sum())

# In[7]:
def describe_categorical(dataset):
    cat_columns = dataset.select_dtypes(include=['object']).columns.tolist()
    
    if len(cat_columns) != 0:
        print('Categorical variables are', cat_columns, '\n' + '==' * 40)
        for cat in cat_columns:
            value_counts = dataset[cat].value_counts()
            proportion = value_counts / len(dataset)
            describe_frame = pd.DataFrame({'Category': value_counts.index, 'Count': value_counts, 'Proportion': proportion})
            describe_frame.reset_index(drop=True, inplace=True)
            print(f"Description of '{cat}':\n", describe_frame, '\n' + '--' * 40)
    else:
        print('There are no categorical variables in the dataset')


# In[8]:
describe_categorical(price_data)

# In[9]:
def describe_numeric(dataset):
    # Select numeric columns
    numeric_columns = dataset.select_dtypes(include=['int64', 'float64'])

    if not numeric_columns.empty:
        print('Numeric variables are:', numeric_columns.columns.tolist(), '\n' + '==' * 40)
        
        # Loop through numeric columns and display statistics
        for column in numeric_columns.columns:
            print(f"Column: {column}")
            print(f"Minimum: {dataset[column].min()}")
            print(f"Maximum: {dataset[column].max()}")
            print(f"Mean: {dataset[column].mean()}")
            print(f"Median: {dataset[column].median()}")
            print("--" * 40)
        
    else:
        print('There are no numeric variables in the dataset')

# In[10]:
describe_numeric(price_data)

# In[15]:
price_data['price_date'] = pd.to_datetime(price_data['price_date'])

# In[16]:
datetime_columns = price_data.select_dtypes(include=['datetime64[ns]'])

if not datetime_columns.empty:
    # Describe datetime columns with datetime_is_numeric=True
    datetime_description = price_data[datetime_columns.columns].describe(datetime_is_numeric=True)
    
    # Create a new DataFrame to store the result
    datetime_table = pd.DataFrame(datetime_description)
    
    # Rename the index column for better readability
    datetime_table.index.name = 'Statistics'
    
    # Reset the index to make 'Statistics' a regular column
    datetime_table.reset_index(inplace=True)
    
    # Display the result as a pandas DataFrame
    print(datetime_table)
else:
    print('There are no datetime variables in the dataset')


# In[17]:
price_data.describe()

# In[18]:
client_data.describe()

# In[20]:
# Visualize the distribution of specific parameters (columns)
columns_to_plot = ['price_off_peak_var', 'price_peak_var', 'price_mid_peak_var',
                    'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix']

plt.figure(figsize=(12, 6))
for col in columns_to_plot:
    sns.histplot(price_data[col], bins=50, kde=True, label=col)
plt.title('Distribution of Price Columns')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[88]:
def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    ax = dataframe.plot(
        kind="bar",
        stacked=True,
        figsize=size_,
        rot=rot_,
        title=title_
    )
    # Define the colors you want to use
    colors = ['blue', 'gray']
    
    # Set the colors for the bars
    for i, bar in enumerate(ax.patches):
        bar.set_color(colors[i % len(colors)])

    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)

    # Rename legend
    plt.legend(["Retention", "Churn"], loc=legend_)

    # Labels
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(),1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
            color=colour,
            size=textsize
        )

def plot_distribution(dataframe, column, ax, bins_=50):
    # Check if 'churn' is available in the DataFrame
    if 'churn' in dataframe.columns:
        temp = pd.DataFrame({"Retention": pd.to_numeric(dataframe[dataframe["churn"] == 0][column], errors='coerce'),
                             "Churn": pd.to_numeric(dataframe[dataframe["churn"] == 1][column], errors='coerce')})
    else:
        # Handle the case where 'churn' column is not available
        temp = pd.DataFrame({column: pd.to_numeric(dataframe[column], errors='coerce')})

    # Remove NaN values
    temp = temp.dropna()

    # Set the gradient colors for the bars
    cmap = plt.cm.get_cmap("viridis")

    # Plot histograms for Retention and Churn groups with custom colors
    if 'Retention' in temp.columns and 'Churn' in temp.columns:
        n, bins, patches = ax.hist([temp["Retention"], temp["Churn"]],
                                   bins=bins_, stacked=True, label=["Retention", "Churn"],
                                   color=[cmap(0.2), cmap(0.8)])
    else:
        n, bins, patches = ax.hist(temp[column],
                                   bins=bins_, label=column,
                                   color=cmap(0.2))

    # Create custom legend patches with gradient colors
    for i, patch in enumerate(patches):
        patch.set_color(cmap(0.2 + i * 0.3))
        
    # X-axis label
    ax.set_xlabel(column)

    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')

# In[41]:
churn = client_data[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")

# In[89]:
rows_of_interest = [5, 10, 15, 20, 25]  # Specify the rows of interest (e.g., rows 5, 10, 15, 20, 25)
columns_of_interest = ['cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons']  # Specify the columns of interest

# Create a single column of histograms for the specified columns and 5 rows
n_rows = len(rows_of_interest)
n_cols = len(columns_of_interest)

fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(9, 15))  # Adjust figsize as needed

# Iterate through rows and plot histograms for each column
for i, row_index in enumerate(rows_of_interest):
    for j, column in enumerate(columns_of_interest):
        plot_distribution(client_data.loc[[row_index], [column]], column, axs[i])

plt.tight_layout()  # Ensure proper layout of subplots
plt.show()


# In[94]:
columns_of_interest = ['cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons']

# Create subplots for each column
n_rows = len(columns_of_interest)
fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(13, 15))

# Iterate through columns and create distribution plots
for i, column in enumerate(columns_of_interest):
    sns.histplot(client_data[column], ax=axs[i], kde=True, bins=30)
    axs[i].set_title(column)
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()  # Ensure proper layout of subplots
plt.show()


# **SUB-TASK 2:**

# In[30]:
# Check for missing values
missing_values = price_data[['price_sensitivity', 'churn']].isnull().sum()

# Check for infinite values
infinite_values = np.isinf(price_data[['price_sensitivity', 'churn']]).sum()

# Print the counts of missing and infinite values
print("Missing Values:")
print(missing_values)

print("\nInfinite Values:")
print(infinite_values)

# In[31]:
# Handle missing values (if any)
price_data.dropna(inplace=True)  # Uncomment this line to remove rows with missing values
# Handle infinite values (if any)
price_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN

# In[32]:
# Calculate the Pearson correlation coefficient and p-value
correlation_coefficient, p_value = stats.pearsonr(price_data['price_sensitivity'], price_data['churn'])

# Set the significance level (α)
alpha = 0.05

# Interpret the results
if p_value < alpha:
    print(f"Correlation coefficient: {correlation_coefficient:.2f}")
    print("There is a statistically significant correlation between price sensitivity and churn.")
else:
    print(f"Correlation coefficient: {correlation_coefficient:.2f}")
    print("There is no statistically significant correlation between price sensitivity and churn.")


# **SUB TASK 3:**

# **From summarize the data insight that I was encompass the distribution analysis, this would be include of:**

# - Churn Analysis: We investigated the relationship between price sensitivity and churn. The correlation analysis revealed a moderate negative correlation (-0.01), in contrast that I had avowal the correlation doesn't significant. This implies that customers who aren't less sensitive to price changes are more likely did not induced to churn.
# - Price Sensitivity: We defined price sensitivity as the ratio of price variance to mean price for each customer. This measure allowed us to identify customers who are more sensitive to price fluctuations.
# - Data Quality: We observed that the dataset contains both numerical and datetime columns. Data quality appears to be generally good, with minimal missing values.
 
# **Suggestions for Data Augmentation:**
# - Customer Demographics: To gain a more comprehensive understanding of churn drivers, it would be valuable to incorporate customer demographics such as age, gender, location, and income. This information can help identify demographic segments with higher churn rates.
# - Customer Interaction Data: Collecting data on customer interactions, such as customer support queries, complaints, or feedback, could provide insights into customer satisfaction and its impact on churn.
# - Competitor Pricing Data: Gathering data on competitor pricing in the energy market would enable a competitive analysis. This data can help assess whether price sensitivity is influenced by the pricing strategies of competitors.
# - Economic Data: Economic indicators such as inflation rates, economic growth, and unemployment rates can affect customer behavior and energy consumption. Incorporating economic data can provide context for analyzing churn.
# - Weather Data: Weather conditions can influence energy consumption patterns. Adding weather data, including temperature and seasonal trends, can enhance predictive models.

# **Survey Data:** Conducting customer surveys to gather feedback on service quality, satisfaction, and reasons for potential churn can provide valuable qualitative insights.
