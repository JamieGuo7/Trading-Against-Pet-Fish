import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime


file_path = './data/tech_data.csv'
raw_data = pd.read_csv(
    file_path,
    parse_dates = ['Date']
)

close_data = raw_data[raw_data['Type'] == 'Close']

# preparing our data for plotting, sorting by date (index)
close_prices = close_data.pivot(
    index = 'Date',
    columns = 'Ticker',
    values = 'Price'
).sort_index()


plt.figure(figsize = (9,15))
close_prices.plot(
    title = 'Weekly Closing Prices for tech stocks over a year',
    xlabel = 'Date',
    ylabel = 'Price (USD)',
    grid = True
)
plt.legend(title = 'Stock', loc = 'upper left')
plt.tight_layout()
plt.show()
plt.close()