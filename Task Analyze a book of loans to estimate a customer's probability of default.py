#!/usr/bin/env python
# coding: utf-8
# # Background Information About the Task
# # Case 3: Display Information
# Loans are an important source of revenue for banks, but they are also associated with the risk that borrowers may default on their loans. A default occurs when a borrower stops making the required payments on a debt.
# The risk team has begun to look at the existing book of loans to see if more defaults should be expected in the future and, if so, what the expected loss will be. They have collected data on customers and now want to build a predictive model that can estimate the probability of default based on customer characteristics. A better estimate of the number of customers defaulting on their loan obligations will allow us to set aside sufficient capital to absorb that loss. They have decided to work with you in the QR team to help predict the possible losses due to the loans that would potentially default in the next year.
# Charlie, an associate in the risk team, who has been introducing you to the business area, sends you a small sample of their loan book and asks if you can try building a prototype predictive model, which she can then test and incorporate into their loss allowances.

# # Case 3: Task Brief
# The risk manager has collected data on the loan borrowers. The data is in tabular format, with each row providing details of the borrower, including their income, total loans outstanding, and a few other metrics. There is also a column indicating if the borrower has previously defaulted on a loan. You must use this data to build a model that, given details for any loan described above, will predict the probability that the borrower will default (also known as PD: the probability of default). Use the provided data to train a function that will estimate the probability of default for a borrower. Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.
# You should produce a function that can take in the properties of a loan and output the expected loss.
# You can explore any technique ranging from a simple regression or a decision tree to something more advanced. You can also use multiple methods and provide a comparative analysis. 
# Submit your code below.

# In[2]:
import pandas as pd
import matplotlib.pyplot as plt

# In[4]:
csv_path = r'C:\Users\Sweet\Downloads\Portofolio\Task 3 and 4_Loan_Data.csv'
data = pd.read_csv(csv_path)
data.sort_index(inplace=True)
data.head()

# In[7]:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Data Preprocessing
data.dropna(inplace=True)  # Drop rows with missing values
X = data[['credit_lines_outstanding', 
          'loan_amt_outstanding', 
          'total_debt_outstanding', 
          'income', 
          'years_employed', 
          'fico_score']]

y = data['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Selection and Training
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate the Probability of Default (PD)
def calculate_pd(customer_info):
    # Customer_info is a dictionary containing the customer's information
    # Customer_info = {'credit_lines_outstanding': 3, 'loan_amt_outstanding': 5000, 
    #'total_debt_outstanding': 8000, 'income': 60000, 'years_employed': 5, 'fico_score': 700}
    pd_value = model.predict_proba(pd.DataFrame(customer_info, index=[0]))[0][1]  # Probability of default (class 1)
    return pd_value

# Calculate the Expected Loss
def calculate_expected_loss(loan_amt, recovery_rate=0.1):
    customer_info = {'credit_lines_outstanding': 3, 
                     'loan_amt_outstanding': loan_amt, 
                     'total_debt_outstanding': loan_amt, 
                     'income': 60000, 
                     'years_employed': 5, 
                     'fico_score': 700}
    pd_value = calculate_pd(customer_info)
    expected_loss = pd_value * loan_amt * (1 - recovery_rate)
    return expected_loss

# Example usage
loan_amount = 10000
expected_loss = calculate_expected_loss(loan_amount)

print(f'Expected Loss for a ${loan_amount} loan: ${expected_loss}')


# The predictive model we developed for estimating the probability of default (PD) has yielded impressive results:
# 
# Accuracy: 98.95%
# Confusion Matrix:
# True Negatives (Non-default): 1647
# False Positives (Predicted default, but didn't): 5
# False Negatives (Predicted non-default, but defaulted): 16
# True Positives (Default): 332
# 
# Our model correctly predicts whether a borrower will default in almost 99% of cases. This high accuracy suggests that the model is effective in assessing default risk. 
# Additionally, let's consider the expected loss for a $10,000 loan with a recovery rate of 10%:

# Expected Loss for a 10,000 loan: $795.35
# This means that, on average, if we were to issue a 10,000 USD loan to a borrower, we can expect an approximate loss of 795.35 USD in the event of a default, taking into account the recovery rate.
