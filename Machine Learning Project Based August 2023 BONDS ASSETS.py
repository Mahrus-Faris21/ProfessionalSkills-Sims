#!/usr/bin/env python
# coding: utf-8

# # Library Supported
# In[112]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.stats as st
from scipy import integrate, stats
from scipy.integrate import quad
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

# # The Principal Interest Model
# In[2]:
#Variable construction
#Certain price
def t_int(start, end):
    dt = 0.01
    t = np.arange(start, end, dt)
    return t

def certain_price(t, T, r):
    t = t_int(t, T)
    if t.sum() == T:
        return 1
    else:
        return np.exp(-(T-t)*r)

# In[3]:
#Equality price from average interest rate
def average_equal(t, T, r):
    t = t_int(t, T)
    return (-(1/(T-t))) * np.log(np.exp(-(T-t)*r))

#Average price interest rate within interval [t, T]
def average_interest(t, T, r):
    av1 = certain_price(t, T, r)
    av2 = average_equal(t, T, r)
    return av1 * av2

# In[4]:
#Forward spot rate, with a certain price respect to 'T'
def spot_rate_forward(t, T, r):
    av3 = certain_price(t, T, r)
    partial_frac_PT = np.gradient(np.log(av3), T)
    return -1 * partial_frac_PT

#Spot rate process, the convenant to reinvest a moment 'T'
def principal_reinvest(t, r, T):
    dt = 0.01
    t_vals = np.arange(t, T, dt)
    integrand_reinvest = [r for s in t_vals]
    results_reinvest = np.trapz(integrand_reinvest, t_vals)
    return np.exp(-results_reinvest)

# In[5]:
#Spot rates in direct account, with no arbitrage

def Spot_Rates_Direct(t, T):
    def integrand(r_s):
        return -r_s

    def conditional_expectation(r_t):
        def integrand_wrapper(r_s):
            return integrand(r_s)

        integral, _ = integrate.quad(integrand_wrapper, t, T)
        return np.exp(-integral)

    expectation, _ = integrate.quad(conditional_expectation, t, T)
    return expectation

#Risky bond market from enduring time 't' besides of investment price moment 't'
def RiskyBond_moment(t, T, n):
    
    def integrand(r_s):
        return r_s

    def conditional_expectation(r_t):
        dt = (T - t) / n
        times = np.linspace(0, t, n+1)
        integrand_values = np.array([integrand(r_s) for r_s in times[:-1]])
        integral_approx = np.sum(integrand_values) * dt
        return np.exp(integral_approx)
    
    expectation = np.sum([conditional_expectation(r_t) for r_t in np.linspace(0, t, n+1)]) / n
    
    return expectation

# # In this fragmention of source code is to recount for Z based simulation

# In[6]
#Random walk
def brownian_distribution(n_steps, T):
    t = T/n_steps
    dt_size = np.sqrt(t)
    dw = np.random.normal(loc=0, scale=dt_size, size=n_steps)
    W = np.cumsum(dw)
    return W

# In[7]:
#Arrange the alpha --> portfolio returns
#Pervade of returns and weights
def weights(market_values):
    total_market = market_values.sum()
    market_weights = market_values/total_market
    return market_weights
    
def xreturns(prices, market_values):
    weights_prices = weights(market_values)
    returns = prices.pct_change().dropna()
    portofolio_returns = (returns * weights_prices).sum().sum()
    return portfolio_returns

def alpha_portfolio(prices, weights_price, market_value):
    returns_x = xreturns(prices, weights_prices)
    xweights = weights(market_value)
    alph_returns = np.dot(returns_x, xweights)
    return alph_returns

def alpha_return_portfolio(market_values, prices):
    total_market = market_values.sum()
    market_weights = market_values/total_market
    market_series = pd.Series(market_weights)
    
    prices_series = pd.Series(prices)
    returns = (prices_series - prices_series.shift(1))/prices_series.shift(1)
    returns = returns.dropna()
    
    portfolio_returns = (returns * market_series.dropna()).sum().sum()
    alpha_returns = np.dot(portfolio_returns, market_weights)
    
    return alpha_returns

#Random Z variable
def Z_input(market_values, prices, t, T, Brownian):
    alpha = alpha_return_portfolio(market_values, prices)
    W_sign = Brownian + alpha * t
    Z_output = np.exp(-alpha * W_sign - 0.50 * alpha**2 * t)
    return Z_output

# # A Girsanov's representation in our source for comtemplated of simulation
# In[8]:
#The sequel Girsanov's Theory in structured of subordinate 2 frame
#Conducting the core algorithm for moment generation function of X* in Multiple returns Lambda (Diffrent drift process)
def lambda_vals(mu, sigma):
    #The X are supposed as interest rate process
    drift_diffusion = lambda t, X, mu, sigma: mu * X + sigma * X
    return drift_diffusion

# In[9]:
#The oflx are pieces of e power to lambda X
def expected_value_oflx(t_vals, X, mu, sigma):
    sum = 0
    lambda_j = lambda_vals(mu, sigma)
    for j in range(1, len(t_vals)):
        sum += (lambda_j(t_vals[j-1], X, mu, sigma)**2) * (t_vals[j] - t_vals[j-1])
    return np.exp(0.50 * sum)

#Calculated expected product in lambda intervals (drift process coefficient)
def prod_value_oflx(lambda_j, t_vals, X):
    product = 1
    lambda_j = lambda_vals(mu, sigma)
    for j in range(1, len(t_vals)):
        product += expected_value_oflx([lambda_j[j-1]], [t_vals[j] - t_vals[j-1]], [X[j] - X[j-1]])
    return product


# In[10]:
def oflx_combined(t_vals, X, mu, sigma):
    sum = 0
    lambda_j = lambda_vals(mu, sigma)
    for j in range(1, len(t_vals)):
        sum += (lambda_j(t_vals[j-1], X, mu, sigma)**2) * (t_vals[j] - t_vals[j-1])
    expected_value = np.exp(0.50 * sum)
    product = 1
    for j in range(1, len(t_vals)):
        product += expected_value
    return product

# # The Sample
# In the purposed to conducted the simulation for this outcomes, I've try to deliberated from the sources of Appendix 5 based. The our of appendix are depict on the type of bond in Indonesian country, then the table have represented as:
# 
# | Bond Type | ORI | SR | SBR |
# | --- | --- | --- | --- |
# | Product type | ORI023-T3<br>ORI023-T6 | SR018T3<br>SR018T5 | SBR012-T2<br>SBR012-T4 |
# | Maturity | 3 years, 15 July 2026<br>6 years, 15 July 2029 | 3 years<br>5 years | 2 years, 10 Feb 2025<br>4 years, 10 Feb 2027 |
# | Determination of Sales Proceeds | 24 July 2023 | - | - |
# | Maximum order | IDR 5 billion (T3)<br>IDR 10 billion (T6) | - | IDR 5 billion (T2)<br>IDR 10 billion (T4) |
# | Minimum order | IDR 1 million and multiples of IDR 1 million | IDR 1 million | IDR 1 million |
# | Issued | 26 July 2023 | 12 May 2023 | 15 February 2023 |
# | Taxes | PPh 10% | PPh 10% | PPh 10% |
# | Coupon/Interest | 5.90% p.a.<br>6.10% p.a. | 6.25% p.a.<br>6.40% p.a. | 6.15% p.a. (BI 7 Day)<br>6.35% p.a. (BI 7 Day) |
# | Early redemption | - | - | 26 Feb – 5 Mar (2024)<br>24 Feb – 4 Mar (2025) |

# For the assumed of initial of times, I cannot to executed with a detailed model i.e. time arrays. Due to a extended count cannot be computed. In the case I would to write from this source:
def time_arrays(start_time, end_time, dt):
    dt = 0.01
    NUM_samples = int((end_time - start_time).total_seconds() / dt)
    time_array = np.linspace(start_time.timestamp(), end_time.timestamp(), NUM_samples)
    time_array = np.array([datetime.datetime.fromtimestamp(ts) for ts in time_array])
    return time_array
# Subsequently, I produced the times intervals with an range (t = 0, T = the depending by the time equals in the samples)

# In[11]:
#I need to potray the times array in eventuate the specific on my sample
t_ORIT3, t_ORIT6, t_SRT3, t_SRT5, t_SBRT2, t_SBRT4 = [0, 0, 0, 0, 0, 0]
T_ORIT3, T_ORIT6, T_SRT3, T_SRT5, T_SBRT2, T_SBRT4 = [3, 6, 3, 5, 2, 4]
C_ORIT3, C_ORIT6, C_SRT3, C_SRT5, C_SBRT2, C_SBRT4 = [0.059, 0.0610, 0.0625, 0.0640, 0.0615, 0.0635]

time_ORIT3 = t_int(t_ORIT3, T_ORIT3)
time_ORIT6 = t_int(t_ORIT6, T_ORIT6)
time_SRT3 = t_int(t_SRT3, T_SRT3)
time_SRT5 = t_int(t_SRT5, T_SRT5)
time_SBRT2 = t_int(t_SBRT2, T_SBRT2)
time_SBRT4 = t_int(t_SBRT4, T_SBRT4)

# # ORI BONDS

# In[12]:
#ORI023
#ORI-T3-T6
#The Sample
Certain_price_ORIT3 = certain_price(t_ORIT3, T_ORIT3, C_ORIT3)
Certain_price_ORIT6 = certain_price(t_ORIT6, T_ORIT6, C_ORIT6)
Average_interest_ORIT3 = average_interest(t_ORIT3, T_ORIT3, C_ORIT3)
Average_interest_ORIT6 = average_interest(t_ORIT6, T_ORIT6, C_ORIT6)
SpotRateForward_ORIT3 = spot_rate_forward(t_ORIT3, T_ORIT3, C_ORIT3)
SpotRateForward_ORIT6 = spot_rate_forward(t_ORIT6, T_ORIT6, C_ORIT6)

#The Persistent Sample
Average_price_ORIT3 = average_equal(t_ORIT3, T_ORIT3, C_ORIT3)
Average_price_ORIT6 = average_equal(t_ORIT6, T_ORIT6, C_ORIT6)

#The Int Induced
Principal_RE_ORIT3 = principal_reinvest(t_ORIT3, C_ORIT3, T_ORIT3)
Principal_RE_ORIT6 = principal_reinvest(t_ORIT6, C_ORIT6, T_ORIT6)
ORIT3_SpotRates_account = Spot_Rates_Direct(t_ORIT3, T_ORIT3)
ORIT6_SpotRates_account = Spot_Rates_Direct(t_ORIT6, T_ORIT6)
RiskyBond_ORIT3 = RiskyBond_moment(t_ORIT3, T_ORIT3, len(Certain_price_ORIT3))
RiskyBond_ORIT6 = RiskyBond_moment(t_ORIT6, T_ORIT6, len(Certain_price_ORIT6))

#The Sample
AvgInterest_Direct_ORIT3 = Average_interest_ORIT3 * ORIT3_SpotRates_account
AvgInterest_Direct_ORIT6 = Average_interest_ORIT6 * ORIT6_SpotRates_account

# In[13]:
#Assuming the market value = Average of interest in intervlals [t, T], and for prices are Certain price
Return_ORIT3_Portfolio = alpha_return_portfolio(Average_interest_ORIT3, Certain_price_ORIT3)
Return_ORIT6_Portfolio = alpha_return_portfolio(Average_interest_ORIT6, Certain_price_ORIT6)

#Modified the random process for return in Z variable brownian
ORIT3_Brownian = brownian_distribution(len(Certain_price_ORIT3), T_ORIT3)
ORIT6_Brownian = brownian_distribution(len(Certain_price_ORIT6), T_ORIT6)
ORIT3_ZDST = Z_input(Average_interest_ORIT3, Certain_price_ORIT3, t_ORIT3, T_ORIT3, Brownian=ORIT3_Brownian)
ORIT6_ZDST = Z_input(Average_interest_ORIT6, Certain_price_ORIT6, t_ORIT6, T_ORIT6, Brownian=ORIT6_Brownian)

#The input for applied Girsanov's theorem
#Sample their included are AvgInterest_Direct_XXX, Average_interest_XXX, and SpotRateForward_XXX
GVS_EXT_ORIT3 = np.log(expected_value_oflx(time_ORIT3, 1, np.mean(AvgInterest_Direct_ORIT3), np.std(AvgInterest_Direct_ORIT3)))
GVS_ORIT3 = np.log(oflx_combined(time_ORIT3, 1, np.mean(AvgInterest_Direct_ORIT3), np.std(AvgInterest_Direct_ORIT3)))

GVS_EXT_ORIT6 = np.log(expected_value_oflx(time_ORIT6, 1, np.mean(AvgInterest_Direct_ORIT6), np.std(AvgInterest_Direct_ORIT6)))
GVS_ORIT6 = np.log(oflx_combined(time_ORIT6, 1, np.mean(AvgInterest_Direct_ORIT6), np.std(AvgInterest_Direct_ORIT6)))


# # SR BONDS

# In[14]:
#SR018
#SR-T3-T5
#The Sample
SRT3_Certain_price = certain_price(t_SRT3, T_SRT3, C_SRT3)
SRT5_Certain_price = certain_price(t_SRT5, T_SRT5, C_SRT5)
SRT3_Average_interest = average_interest(t_SRT3, T_SRT3, C_SRT3)
SRT5_Average_interest = average_interest(t_SRT5, T_SRT5, C_SRT5)
SRT3_SpotRateForward = spot_rate_forward(t_SRT3, T_SRT3, C_SRT3)
SRT5_SpotRateForward = spot_rate_forward(t_SRT5, T_SRT5, C_SRT5)

# The Persistent Sample
SRT3_Average_price = average_equal(t_SRT3, T_SRT3, C_SRT3)
SRT5_Average_price = average_equal(t_SRT5, T_SRT5, C_SRT5)

# The Int Induced
SRT3_Principal_RE = principal_reinvest(t_SRT3, C_SRT3, T_SRT3)
SRT5_Principal_RE = principal_reinvest(t_SRT5, C_SRT5, T_SRT5)
SRT3_SpotRates_account = Spot_Rates_Direct(t_SRT3, T_SRT3)
SRT5_SpotRates_account = Spot_Rates_Direct(t_SRT5, T_SRT5)
SRT3_RiskyBond = RiskyBond_moment(t_SRT3, T_SRT3, len(SRT3_Certain_price))
SRT5_RiskyBond = RiskyBond_moment(t_SRT5, T_SRT5, len(SRT5_Certain_price))

#The Sample
SRT3_AvgInterest_Direct = SRT3_Average_interest * SRT3_SpotRates_account
SRT5_AvgInterest_Direct = SRT5_Average_interest * SRT5_SpotRates_account

# In[15]:
# Assuming the market value = Average of interest in intervals [t, T], and for prices are Certain price
Return_SRT3_Portfolio = alpha_return_portfolio(SRT3_Average_interest, SRT3_Certain_price)
Return_SRT5_Portfolio = alpha_return_portfolio(SRT5_Average_interest, SRT5_Certain_price)

# Modified the random process for return in Z variable brownian
SRT3_Brownian = brownian_distribution(len(SRT3_Certain_price), T_SRT3)
SRT5_Brownian = brownian_distribution(len(SRT5_Certain_price), T_SRT5)
SRT3_ZDST = Z_input(SRT3_Average_interest, SRT3_Certain_price, t_SRT3, T_SRT3, Brownian=SRT3_Brownian)
SRT5_ZDST = Z_input(SRT5_Average_interest, SRT5_Certain_price, t_SRT5, T_SRT5, Brownian=SRT5_Brownian)

# The input for applied Girsanov's theorem
# Sample their included are AvgInterest_Direct_XXX, Average_interest_XXX, and SpotRateForward_XXX
GVS_EXT_SRT3 = np.log(expected_value_oflx(time_SRT3, 1, np.mean(SRT3_AvgInterest_Direct), np.std(SRT3_AvgInterest_Direct)))
GVS_SRT3 = np.log(oflx_combined(time_SRT3, 1, np.mean(SRT3_AvgInterest_Direct), np.std(SRT3_AvgInterest_Direct)))

GVS_EXT_SRT5 = np.log(expected_value_oflx(time_SRT5, 1, np.mean(SRT5_AvgInterest_Direct), np.std(SRT5_AvgInterest_Direct)))
GVS_SRT5 = np.log(oflx_combined(time_SRT5, 1, np.mean(SRT5_AvgInterest_Direct), np.std(SRT5_AvgInterest_Direct)))


# # SBR BONDS

# In[16]:
#SBR012
#SBR-T2-T4
#The Sample
SBRT2_Certain_price = certain_price(t_SBRT2, T_SBRT2, C_SBRT2)
SBRT4_Certain_price = certain_price(t_SBRT4, T_SBRT4, C_SBRT4)
SBRT2_Average_interest = average_interest(t_SBRT2, T_SBRT2, C_SBRT2)
SBRT4_Average_interest = average_interest(t_SBRT4, T_SBRT4, C_SBRT4)
SBRT2_SpotRateForward = spot_rate_forward(t_SBRT2, T_SBRT2, C_SBRT2)
SBRT4_SpotRateForward = spot_rate_forward(t_SBRT4, T_SBRT4, C_SBRT4)

# The Persistent Sample
SBRT2_Average_price = average_equal(t_SBRT2, T_SBRT2, C_SBRT2)
SBRT4_Average_price = average_equal(t_SBRT4, T_SBRT4, C_SBRT4)

# The Int Induced
SBRT2_Principal_RE = principal_reinvest(t_SBRT2, C_SBRT2, T_SBRT2)
SBRT4_Principal_RE = principal_reinvest(t_SBRT4, C_SBRT4, T_SBRT4)
SBRT2_SpotRates_account = Spot_Rates_Direct(t_SBRT2, T_SBRT2)
SBRT4_SpotRates_account = Spot_Rates_Direct(t_SBRT4, T_SBRT4)
SBRT2_RiskyBond = RiskyBond_moment(t_SBRT2, T_SBRT2, len(SBRT2_Certain_price))
SBRT4_RiskyBond = RiskyBond_moment(t_SBRT4, T_SBRT4, len(SBRT4_Certain_price))

#The Sample
SBRT2_AvgInterest_Direct = SBRT2_Average_interest * SBRT2_SpotRates_account
SBRT4_AvgInterest_Direct = SBRT4_Average_interest * SBRT4_SpotRates_account

# In[17]:
# Assuming the market value = Average of interest in intervals [t, T], and for prices are Certain price
Return_SBRT2_Portfolio = alpha_return_portfolio(SBRT2_Average_interest, SBRT2_Certain_price)
Return_SBRT4_Portfolio = alpha_return_portfolio(SBRT4_Average_interest, SBRT4_Certain_price)

# Modified the random process for return in Z variable brownian
SBRT2_Brownian = brownian_distribution(len(SBRT2_Certain_price), T_SBRT2)
SBRT4_Brownian = brownian_distribution(len(SBRT4_Certain_price), T_SBRT4)
SBRT2_ZDST = Z_input(SBRT2_Average_interest, SBRT2_Certain_price, t_SBRT2, T_SBRT2, Brownian=SBRT2_Brownian)
SBRT4_ZDST = Z_input(SBRT4_Average_interest, SBRT4_Certain_price, t_SBRT4, T_SBRT4, Brownian=SBRT4_Brownian)

# The input for applied Girsanov's theorem
# Sample their included are AvgInterest_Direct_XXX, Average_interest_XXX, and SpotRateForward_XXX
GVS_EXT_SBRT2 = np.log(expected_value_oflx(time_SBRT2, 1, np.mean(SBRT2_AvgInterest_Direct), np.std(SBRT2_AvgInterest_Direct)))
GVS_SBRT2 = np.log(oflx_combined(time_SBRT2, 1, np.mean(SBRT2_AvgInterest_Direct), np.std(SBRT2_AvgInterest_Direct)))

GVS_EXT_SBRT4 = np.log(expected_value_oflx(time_SBRT4, 1, np.mean(SBRT4_AvgInterest_Direct), np.std(SBRT4_AvgInterest_Direct)))
GVS_SBRT4 = np.log(oflx_combined(time_SBRT4, 1, np.mean(SBRT4_AvgInterest_Direct), np.std(SBRT4_AvgInterest_Direct)))


# # The Data Sample with Delineate of Stats
# In bellow from this code as made it to explain the set of data, that I will demonstrate the 'int' type output and generates the descriptive statistics from. Subsequently, the set of each table at the 'int' variety are used either to visualize or to perform displayed in the dataset only.

# In[50]:
#Summarization of Data
Bonds_input = {
    
    'Bond':['ORI_T3', 'ORI_T6', 'SR_T3', 'SR_T5', 'SBR_T2', 'SBR_T4'],
    'T':[T_ORIT3, T_ORIT6, T_SRT3, T_SRT5, T_SBRT2, T_SBRT4],
    't': [t_ORIT3, t_ORIT6, t_SRT3, t_SRT5, t_SBRT2, t_SBRT4],
    'Coupond_Bonds': [0.0590, 0.0610, 0.0625, 0.0640, 0.0615, 0.0635],
    'Principal Reinvest': [Principal_RE_ORIT3, Principal_RE_ORIT6, SRT3_Principal_RE, SRT5_Principal_RE,
                          SBRT2_Principal_RE, SBRT4_Principal_RE],
    'Direct Spot Rates': [ORIT3_SpotRates_account, ORIT6_SpotRates_account, SRT3_SpotRates_account, SRT5_SpotRates_account,
                          SBRT2_SpotRates_account, SBRT4_SpotRates_account],
    'Risky Bonds Params': [RiskyBond_ORIT3, RiskyBond_ORIT6, SRT3_RiskyBond, SRT5_RiskyBond, SBRT2_RiskyBond, SBRT4_RiskyBond],
    'Random Expectation': [GVS_EXT_ORIT3, GVS_EXT_ORIT6, GVS_EXT_SRT3, GVS_EXT_SRT5, GVS_EXT_SBRT2, GVS_EXT_SBRT4],
    'Total Expectation': [GVS_ORIT3, GVS_ORIT6, GVS_SRT3, GVS_SRT5, GVS_SBRT2, GVS_SBRT4]
    
}
Bonds_data = pd.DataFrame(Bonds_input)

Bonds_data


# In[19]:
Bonds_data.describe().transpose()

# Now, at the subordinates of this code is to recount the depict on each function, that I have completion in the section previous of any Bond type. The code are tells to executed the descriptive statistics from pervade as it was to serves.

# In[20]:
def descriptive_stats(my_array):
    mean = np.mean(my_array)
    median = np.median(my_array)
    minimum = np.min(my_array)
    maximum = np.max(my_array)
    variance = np.var(my_array)
    std_dev = np.std(my_array)
    skewness = stats.skew(my_array)
    kurtosis = stats.kurtosis(my_array)
    
    return {'Mean': mean, 
            'Median': median, 
            'Minimum': minimum, 
            'Maximum': maximum, 
            'Variance': variance, 
            'Standard deviation': std_dev, 
            'Skewness': skewness, 
            'Kurtosis': kurtosis}

# The precise in accordance of this code is a certain example in which the variable of previous that I have written.

# In[21]:
descriptive_stats(Certain_price_ORIT3)

# # Statistics Testing

# To adjusted the model statistical testing, I would to prefer in the manuscript that I wrap up in their Chapter 3 (Method Analysis/Tools Analysis), from the fragment of the section I write the tools to analyze the value in the umpteen sets of data, I choose to calculating the sample with T-Test Distribution and ANOVA from two-sample test. Afterwards, I genuinely will potray the results into the graph stats from corresponding to the aim the script before.
# Additional information for T-Test Dist., the two-sample in mentioned is the X1: Certain price and Forward spot rate, while the X2: Average interest price interval with multiplication of Spot rates direct in account investor.
# In the other hand, ANOVA samples will employeed for both two variable as the piece predicates in previous description.

# In[22]:
stats.shapiro(Certain_price_ORIT3)

# In[23]:
stats.levene(Certain_price_ORIT3, AvgInterest_Direct_ORIT3)

# Due to theorem of Central Limit, it is possible to accepted the type of distribution, while the results in the above are p < 0.05, then as the sample population > more that was savvy to t-test can be used to the null hypothesis. Moreover, it means of two samples are equal. 

# In[24]:
stats.ttest_ind(Certain_price_ORIT3, AvgInterest_Direct_ORIT3, equal_var=False)

# The turns up results as the authentically negative, this would expected from those distributions are being different in unequal mean, and futhermore the data would significanlly proper a < p(5%). In the case, this cannot be identify to state as the same meaning, that the data are would variety in temporal of times intervals [t, T].
# # Set A Dataset (With the Example for Bond ORI & SRT)

# In[56]:
AAX = pd.DataFrame(
    {
        'Bonds Type': np.repeat(['ORI023', 'SRT018'], 300),
        'Certain Price': np.concatenate([Certain_price_ORIT3, 
                                         SRT3_Certain_price]),
        'Spot Rate Forward': np.concatenate([SpotRateForward_ORIT3, 
                                             SRT3_SpotRateForward]),
        'Average Direct Account': np.concatenate([AvgInterest_Direct_ORIT3, 
                                                  SRT3_AvgInterest_Direct])
    }
)
AAX.head()

# To sum it up, the 'AAX' DataFrame is made to organize and store data about two types of bonds, which includes their certain prices, spot rates, and average direct account values. We use the 'head()' function to show the first few rows of this DataFrame and get an initial look at the data

# In[23]:
random_data = np.random.normal(0, 2, size=(len(AAX), 3))
new_AAX = pd.DataFrame(AAX[['Certain Price', 'Spot Rate Forward', 'Average Direct Account']].values + random_data, columns=['Certain Price', 'Spot Rate Forward', 'Average Direct Account'])
new_AAX['Bonds Type'] = AAX['Bonds Type']

col = new_AAX.pop('Bonds Type')
new_AAX.insert(0, col.name, col)
new_AAX.head()

# This code introduces random variations to particular columns in the 'AAX' DataFrame. It keeps the 'Bonds Type' column, reorganizes the columns to place 'Bonds Type' at the beginning, and then displays the modified DataFrame. It seems the code's goal is to add randomness to specific financial data while retaining information about the bond types for analysis or simulations

# In[24]:
this code adds random noise to specific columns of the "AAX" DataFrame, retains the 'Bonds Type' column, and rearranges the columns to make 'Bonds Type' the first column in the resulting DataFrame, which is then displayed. The purpose of this code appears to be to introduce variability into certain financial data while preserving the bond type information for analysis or simulation purposes.# fit the linear regression model
from scipy.stats import f_oneway

for col in new_AAX.columns[1:]:
    groups = [new_AAX[col][new_AAX['Bonds Type'] == bond_type] for bond_type in new_AAX['Bonds Type'].unique()]
    f_statistic, p_value = f_oneway(*groups)
    print(f'{col}: F-statistic={f_statistic:.2f}, p-value={p_value:.4f}')

# We performed an ANOVA analysis on the 'Certain Price,' 'Spot Rate Forward,' and 'Average Direct Account' columns of the AAX DataFrame. We calculated the F-statistic and p-values for each column. For 'Certain Price' and 'Spot Rate Forward,' the F-statistic values were 2.00 and 0.11, respectively, with corresponding p-values of 0.1575 and 0.7447. As for 'Average Direct Account,' the F-statistic was 79.59, and the p-value was 0.0000. These results suggest that the 'Average Direct Account' column significantly influences the recovery, while the other columns have less impact.
# # Creating the Graph Scheme

# Stage 1: Scatter & Regression Analysis
# Initial stage: Create a scatter plot and regression analysis to examine how 'Certain Price' and 'Spot Rate Forward' affect 'Average Interest Rate.'

# Stage 2: Data Exploration Plot
# Next, visualize 'Bonds_Data' with a plot to reveal key data trends or relationships.

# Stage 3: Boxplot Analysis
# Third stage: utilize boxplots with two independent and one dependent variable to understand data distribution and outliers.

# Stage 4: 2D Plot - Random Variable Z vs. Market Value
# Lastly, create a 2D plot illustrating how 'Random Variable Z' influences 'Market Value' variations, enhancing analysis insights.

# In[26]:
fig, ax = plt.subplots(1, 2, figsize=(12, 3))

g1 = sns.regplot(
            x="Certain Price",
            y="Average Direct Account",
            scatter_kws={"color": "blue", "s": 5}, 
            line_kws={"color": "red"},
            marker = "o",
            data=new_AAX,
            ax=ax[0]
          ) 
g2 = sns.regplot(
            x="Spot Rate Forward",
            y="Average Direct Account",
            scatter_kws={"color": "blue", "s": 5}, 
            line_kws={"color": "red"},
            marker = "x",
            data=new_AAX,
            ax=ax[1]
          ) 

ax[0].set_title("Bonds Type: ORI023-T3 & SR018-T3")
ax[1].set_title("Bonds Type: ORI023-T3 & SR018-T3")

plt.tight_layout()
plt.show()

# In[69]:
sns.lmplot(x="Certain Price", 
           y="Average Direct Account", 
           data=new_AAX, 
           order=2, 
           scatter_kws={
               "color": "black", 
               "s":5
           }, 
           line_kws={
               "color": "red", 
               "label": "Order 2 Regression Line"
           }, 
           ci=95
          )
sns.regplot(x="Certain Price", 
            y="Average Direct Account", 
            data=new_AAX, 
            scatter_kws={
                "color": "black", 
                "s":5
            }, 
            line_kws={
                "color": "blue", 
                "label": "Linear Regression Line"
            }
           )
sns.set_style("white")
plt.legend() 
plt.title("Scatter Plot with Regression Line from ORI03-T3 & SR018-T3")

# In[38]:
df = pd.DataFrame(data=[(0.0590,Principal_RE_ORIT3,ORIT3_SpotRates_account,RiskyBond_ORIT3), 
                        (0.0610,Principal_RE_ORIT6,ORIT6_SpotRates_account,RiskyBond_ORIT6), 
                        (0.0625,SRT3_Principal_RE,SRT3_SpotRates_account,SRT3_RiskyBond), 
                        (0.0640,SRT5_Principal_RE,SRT5_SpotRates_account,SRT5_RiskyBond), 
                        (0.0615,SBRT2_Principal_RE,SBRT2_SpotRates_account,SBRT2_RiskyBond), 
                        (0.0635,SBRT4_Principal_RE,SBRT4_SpotRates_account,SBRT4_RiskyBond)],
                  index=['ORI T3', 'ORI T6', 'SR T3', 'SR T5', 'SBR T2', 'SBR T4'], 
                  columns=['Coupond Bond', 'Principal Reinvestment', 'Direct Spot Rates', 'Risky Bonds'])
df.head()

# In[22]:
ax =  df.plot.bar(y='Principal Reinvestment', ylabel='Principal Reinvestment', figsize=(8, 5), color='black')
df.plot(y='Risky Bonds', ax=ax, label='Risky Bonds', use_index=False, secondary_y=True, mark_right=False, color='red')
ax.right_ax.set_ylabel('Risky Bonds')
ax.legend(loc='upper right', bbox_to_anchor=(-0.01, 1.0))
ax.set_title('Investment Comparison')

# In[118]:
get_ipython().run_line_magic('matplotlib', 'inline')
group_pie = ['ORI T3', 'ORI T6', 'SR T3', 'SR T5', 'SBR T2', 'SBR T4']
pie_set = pd.Series([0.0590, 0.0610, 0.0625, 0.0640, 0.0615, 0.0635],
               index=['ORI T3', 'ORI T6', 'SR T3', 'SR T5', 'SBR T2', 'SBR T4'] 
               )

explode = (0, 0.1, 0, 0, 0, 0)
colors = ['#191970', '#001CF0', '#0038E2', '#0055D4', '#0071C6', '#008DB8']
font = {'family': 'serif', 'size': 14, 'weight': 'bold'}
fig, ax = plt.subplots(figsize=(8, 5))
pie_wedge, pie_label, pie_percent = ax.pie(pie_set, labels=pie_set.index, colors=colors, 
                                           explode=explode, autopct='%1.1f%%', 
                                           wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
for text in pie_label:
    text.set_fontsize(14)
    text.set_fontweight('bold')
    text.set_fontfamily('serif')
    
plt.axis('equal')
plt.ylabel('')
plt.legend(labels=pie_set.index, loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()

# In[85]:
N = 300
Certain_price = np.concatenate([Certain_price_ORIT3, SRT3_Certain_price])
SpotRateForward = np.concatenate([SpotRateForward_ORIT3, SRT3_SpotRateForward])
AvgInterest_Direct = np.concatenate([AvgInterest_Direct_ORIT3, SRT3_AvgInterest_Direct])
Bonds_Type = np.repeat(['ORI023', 'SRT018'], N)

df = pd.DataFrame({
    'Certain Price': Certain_price,
    'Spot Rate Forward': SpotRateForward,
    'Average Direct Account': AvgInterest_Direct,
    'Bonds Type': Bonds_Type
})
fig, ax = plt.subplots(figsize=(9, 5))
boxprops = {'color': 'blue'}
whiskerprops = {'color': 'red'}
capprops = {'color': 'red'}
ax = df.boxplot(column='Certain Price', by='Bonds Type', showfliers=True, 
                positions=range(df['Bonds Type'].unique().shape[0]),
                boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, ax=ax)
ax.set_facecolor('white')
sns.pointplot(x='Bonds Type', y='Certain Price', data=df.groupby('Bonds Type', as_index=False).mean(), ax=ax, color='green')

# In[101]:
fig, ax = plt.subplots(figsize=(9, 5))
boxprops = {'color': 'blue'}
whiskerprops = {'color': 'red'}
capprops = {'color': 'red'}
ax = df.boxplot(column='Average Direct Account', by='Bonds Type', showfliers=True, 
                positions=range(df['Bonds Type'].unique().shape[0]),
                boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, ax=ax)
ax.set_facecolor('white')
sns.pointplot(x='Bonds Type', y='Average Direct Account', data=df.groupby('Bonds Type', as_index=False).mean(), ax=ax, color='green')

# In[141]:
# create a figure with 6 subplots
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10, 20))

# plot each X and Y variable on a separate subplot
axs[0].plot(Certain_price_ORIT3, ORIT3_ZDST, label='ORIT3', color='b')
axs[0].set_xlabel('Certain Price ORIT3')
axs[0].set_ylabel('ORIT3 ZDST')
axs[0].ticklabel_format(style='plain', axis='both')
axs[0].axhline(y=np.mean(ORIT3_ZDST), color='r', linestyle='--', label='Mean Z')
axs[0].axvline(x=np.median(Certain_price_ORIT3), color='k', linestyle='--', label='Median Certain Price')

axs[1].plot(Certain_price_ORIT6, ORIT6_ZDST, label='ORIT6', color='b')
axs[1].set_xlabel('Certain Price ORIT6')
axs[1].set_ylabel('ORIT6 ZDST')
axs[1].ticklabel_format(style='plain', axis='both')
axs[1].axhline(y=np.mean(ORIT6_ZDST), color='r', linestyle='--', label='Mean Z')
axs[1].axvline(x=np.median(Certain_price_ORIT6), color='k', linestyle='--', label='Median Certain Price')

axs[2].plot(SRT3_Certain_price, SRT3_ZDST, label='SRT3', color='b')
axs[2].set_xlabel('Certain Price SRT3')
axs[2].set_ylabel('SRT3 ZDST')
axs[2].ticklabel_format(style='plain', axis='both')
axs[2].axhline(y=np.mean(SRT3_ZDST), color='r', linestyle='--', label='Mean Z')
axs[2].axvline(x=np.median(SRT3_Certain_price), color='k', linestyle='--', label='Median Certain Price')

axs[3].plot(SRT5_Certain_price, SRT5_ZDST, label='SRT5', color='b')
axs[3].set_xlabel('Certain Price SRT5')
axs[3].set_ylabel('SRT5 ZDST')
axs[3].ticklabel_format(style='plain', axis='both')
axs[3].axhline(y=np.mean(SRT5_ZDST), color='r', linestyle='--', label='Mean Z')
axs[3].axvline(x=np.median(SRT5_Certain_price), color='k', linestyle='--', label='Median Certain Price')

axs[4].plot(SBRT2_Certain_price, SBRT2_ZDST, label='SBRT2', color='b')
axs[4].set_xlabel('Certain Price SBRT2')
axs[4].set_ylabel('SBRT2 ZDST')
axs[4].ticklabel_format(style='plain', axis='both')
axs[4].axhline(y=np.mean(SBRT2_ZDST), color='r', linestyle='--', label='Mean Z')
axs[4].axvline(x=np.median(SBRT2_Certain_price), color='k', linestyle='--', label='Median Certain Price')

axs[5].plot(SBRT4_Certain_price, SBRT4_ZDST, label='SBRT4', color='b')
axs[5].set_xlabel('Certain Price SBRT4')
axs[5].set_ylabel('SBRT4 ZDST')
axs[5].ticklabel_format(style='plain', axis='both')
axs[5].axhline(y=np.mean(SBRT4_ZDST), color='r', linestyle='--', label='Mean Z')
axs[5].axvline(x=np.median(SBRT4_Certain_price), color='k', linestyle='--', label='Median Certain Price')
# adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# set the background color to white
fig.patch.set_facecolor('white')
for ax in axs:
    ax.set_facecolor('white')
    ax.legend(loc='best')

# display the plot
plt.show()

# In all the examples we've seen, the calculations are ongoing. That's why this section is primarily used for creating and identifying output formats in research projects related to bonds
