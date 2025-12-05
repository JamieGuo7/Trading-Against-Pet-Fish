import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Getting data
tech = ['GOOG', 'MSFT', 'AAPL', 'NVDA', 'AMZN']
data = yf.download(tech, period = '1y', interval='1wk')

# Cleaning and Structuring
data.dropna(inplace = True)
