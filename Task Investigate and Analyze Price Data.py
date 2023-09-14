#!/usr/bin/env python
# coding: utf-8

# # Background Information About the Task

# #  Case 1: Display Information

# You are a quantitative researcher working with a commodity trading desk. Alex, a VP on the desk, wants to start trading natural gas storage contracts. However, the available market data must be of higher quality to enable the instrument to be priced accurately. They have sent you an email asking you to help extrapolate the data available from external feeds to provide more granularity, considering seasonal trends in the price as it relates to months in the year. To price the contract, we will need historical data and an estimate of the future gas price at any date.
# 
# Commodity storage contracts represent deals between warehouse (storage) owners and participants in the supply chain (refineries, transporters, distributors, etc.). The deal is typically an agreement to store an agreed quantity of any physical commodity (oil, natural gas, agriculture) in a warehouse for a specified amount of time. The key terms of such contracts (e.g., periodic fees for storage, limits on withdrawals/injections of a commodity) are agreed upon inception of the contract between the warehouse owner and the client. The injection date is when the commodity is purchased and stored, and the withdrawal date is when the commodity is withdrawn from storage and sold. More details can be found here: Understanding Commodity Storage.
# 
# A client could be anyone who would fall within the commodities supply chain, such as producers, refiners, transporters, and distributors. This group would also include firms (commodities trading, hedge funds, etc.) whose primary aim is to take advantage of seasonal or intra-day price differentials in physical commodities. For example, if a firm is looking to buy physical natural gas during summer and sell it in winter, it would take advantage of the seasonal price differential mentioned above. The firm would need to leverage the services of an underground storage facility to store the purchased inventory to realize any profits from this strategy.

# # Case 1: Task Brief

# After asking around for the source of the existing data, you learn that the current process is to take a monthly snapshot of prices from a market data provider, which represents the market price of natural gas delivered at the end of each calendar month. This data is available for roughly the next 18 months and is combined with historical prices in a time series database. After gaining access, you are able to download the data in a CSV file.
# 
# You should use this monthly snapshot to produce a varying picture of the existing price data, as well as an extrapolation for an extra year, in case the client needs an indicative price for a longer-term storage contract.
# 
# 1. Download the monthly natural gas price data.
# 2. Each point in the data set corresponds to the purchase price of natural gas at the end of a month, from 31st October 2020 to 30th September 2024.
# 3. Analyze the data to estimate the purchase price of gas at any date in the past and extrapolate it for one year into the future. 
# 4. Your code should take a date as input and return a price estimate.
# 5. Try to visualize the data to find patterns and consider what factors might cause the price of natural gas to vary. This can include looking at months of the year for seasonal trends that affect the prices, but market holidays, weekends, and bank holidays need not be accounted for. Submit your completed code below.
# 
# Note: This role often requires the knowledge and utilization of data analysis and machine learning. Python is a useful tool and one that JPMorgan Chase uses a lot in quantitative research since it’s capable of completing complex tasks.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


csv_path = r'C:\Users\Sweet\Downloads\Portofolio\Nat_Gas.csv'


# In[3]:


price_data = pd.read_csv(csv_path, parse_dates=['Dates'], index_col='Dates')


# In[4]:


price_data.sort_index(inplace=True)
price_data.head()


# In[11]:


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

# Plotting the Natural Gas Prices Over Time
ax.plot(price_data.index, price_data['Prices'], marker='o', linestyle='-', color='b')

ax.set_title('Natural Gas Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.grid(True)

plt.show()


# # The Answers

# Analysis of Natural Gas Storage Contract Pricing Trends:
# 
# The provided line charts offer a valuable perspective on historical and future trends in natural gas prices. These insights are crucial for accurately pricing natural gas storage contracts, particularly considering seasonal variations.
# 
# 1. Historical Price Trends (2021-2022):
# - Examining the data for the years 2021 to 2022, we observe distinct price dynamics, with a focus on the month of July.
# - During this period, prices exhibited noticeable fluctuations, initially trending downward and subsequently experiencing a gradual ascent.
# - In July 2021, we noted a modest price increase, while July 2022 witnessed a more substantial yet still relatively moderate rise.
# 
# 2. Contrasting Behaviors (January 2022 vs. January 2023):
# 
# - A striking observation is the contrasting behavior between January 2022 and January 2023.
# - January 2022 featured a sharp upward surge in prices, suggesting potential trading opportunities.
# - However, it's essential to recognize that January 2023 exhibited different price dynamics, emphasizing the importance of considering both historical and forward-looking data.
# 
# 3. Positive Prospects in 2023 Futures:
# 
# - An encouraging aspect revealed by the charts is the positive outlook for natural gas prices in 2023.
# - The data suggests a period of prosperity and potential for substantial earnings.
# - Prices are thriving at elevated levels, indicating a bullish market sentiment.
# - Additionally, the stability of these prices suggests the potential for sustained high value.
# 
# 4. Anticipated Price Decline (July 2024):
# 
# - While the outlook for 2023 appears favorable, it's prudent to note an anticipated price decline in July 2024.
# - This forecasted decrease may present an opportunity for traders seeking to enter the market at a lower price point.
# - In conclusion, the analysis of these line charts provides valuable insights for pricing natural gas storage contracts. It underscores the significance of considering historical trends, recognizing divergent behaviors, and leveraging opportunities in the futures market. Traders should exercise due diligence and employ comprehensive analysis before making trading decisions in the dynamic natural gas market.

# Analyzing Natural Gas Storage Contract Pricing Trends for Informed Decision-Making:
# 
# In response to Alex's request to assist in trading natural gas storage contracts, it's crucial to understand the significance of high-quality market data, particularly when considering seasonal price trends and estimating future gas prices.
# 
# 1. Leveraging Historical Data:
# 
# - To effectively price natural gas storage contracts, we must begin by leveraging historical data. This data will provide essential insights into past price movements, enabling us to make more informed decisions.
# - Historical data helps us recognize patterns and trends, such as fluctuations in gas prices during specific months, which can be highly valuable for contract pricing.
# 
# 2. Understanding Commodity Storage Contracts:
# 
# - Commodity storage contracts, such as those for natural gas, are agreements between storage facility owners and participants in the supply chain, including refineries, transporters, distributors, and more.
# - These contracts involve storing a specified quantity of a physical commodity for a predetermined duration.
# - Key terms, including storage fees and withdrawal/injection limits, are established at the contract's inception.
# 
# 3. The Role of Storage Facilities:
# 
# - Storage facilities play a pivotal role in enabling various participants, including producers, refiners, and traders, to manage their commodity inventories effectively.
# - For example, consider a firm aiming to capitalize on seasonal price differentials by purchasing natural gas in summer and selling it in winter. This strategy relies on the availability of underground storage facilities to store inventory during off-peak seasons.
# 
# 4. Consideration of Seasonal Price Trends:
# 
# - As part of our data analysis, we should closely examine seasonal price trends specific to natural gas.
# - These trends may reveal opportunities for strategic trading, such as entering the market during periods of lower prices and storing gas for later sale during periods of higher demand and prices.
# 
# 5. Estimating Future Gas Prices:
# 
# - In addition to historical data, estimating future gas prices is essential for pricing storage contracts accurately.
# - Advanced forecasting models and market analysis can provide us with estimates of future prices at any given date.
# - Such estimates are critical for determining the potential profitability of storage contracts over time.
# 
# 6. Historical Price Trends (2021-2022):
# 
# - Examining the data, we can discern historical price trends, particularly during 2021 and 2022.
# - In these years, we observed fluctuations in prices, notably in the month of July. Prices initially decreased, followed by a gradual rise.
# - July 2021 showed a modest increase in prices, while July 2022 witnessed a more substantial yet still relatively moderate rise.
# 
# 7. Contrasting Behaviors (January 2022 vs. January 2023):
# 
# - A striking observation is the contrasting behavior between January 2022 and January 2023.
# - January 2022 featured a sharp upward surge in prices, suggesting potential trading opportunities.
# - However, it's essential to recognize that January 2023 exhibited different price dynamics, emphasizing the importance of considering both historical and forward-looking data.
# 
# 8. Positive Prospects in 2023 Futures:
# 
# - An encouraging aspect revealed by the charts is the positive outlook for natural gas prices in 2023.
# - The data suggests a period of prosperity and potential for substantial earnings.
# - Prices are thriving at elevated levels, indicating a bullish market sentiment.
# - Additionally, the stability of these prices suggests the potential for sustained high value.
# 
# 9. Anticipated Price Decline (July 2024):
# 
# - While the outlook for 2023 appears favorable, it's prudent to note an anticipated price decline in July 2024.
# - This forecasted decrease may present an opportunity for traders seeking to enter the market at a lower price point.
# - In conclusion, by harnessing historical data, understanding the dynamics of commodity storage contracts, and considering seasonal trends, we can enhance our ability to price natural gas storage contracts accurately. This analysis is essential for participants across the commodities supply chain and firms seeking to optimize their trading strategies based on price differentials and storage opportunities.
