import numpy as np
import pandas as pd


"""

Calculates relevant indicators for each stock, saving this and storing it as a csv file for future use.

"""


# Download as a dataframe
file_path = '../../data/ESGU_data.csv'
df = pd.read_csv(file_path)

df = df.sort_values(['Ticker', 'Date'])

# Calculate moving averages



