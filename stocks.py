import pandas as pd
import glob
import numpy as np
import datetime
import os


path = 'stocks/'
all_files = glob.glob(os.path.join(path, "*.csv"))

df = pd.concat((pd.read_csv(f, index_col=None, header=0) for f in all_files), axis=0, ignore_index=False, sort=True)


print(df.head())

print(df[['Name', 'Symbol']].head())

print(list(df.columns))

print(df[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6',
              'Unnamed: 7', 'Unnamed: 8']].head())

df = df.dropna(subset=['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6',
              'Unnamed: 7', 'Unnamed: 8'])

print(df[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6',
              'Unnamed: 7', 'Unnamed: 8']].head())


#probably can just drop these columns.  not sure how/why they got made?

df = df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6',
              'Unnamed: 7', 'Unnamed: 8'], axis=1)

print("Unorganized")
print(list(df.columns))
print()
#now I want the columns back in the correct order

stocks = df[['Name', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Net Chg', '% Chg', 'Volume', '52 Wk High', '52 Wk Low',
             'Div', 'Yield', 'P/E', 'YTD % Chg']].copy()
print("Stocks: ")
print(list(stocks.columns))
