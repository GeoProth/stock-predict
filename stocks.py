from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import glob
import numpy as np
import datetime
import os

pd.set_option('display.max_columns', 18)
pd.set_option('display.width', 1000)

# File Path
filenames = glob.glob('stocks/*.csv')


def import_csv(csv_path):
    missing_values = ['na', '...'] #data has missing values as ... (must replace)
    df = pd.read_csv(csv_path, na_values=missing_values)
    index = csv_path.find('-')
    date = csv_path[index+1:-4]
    day = date[:date.find('-')]
    date = date[date.find('-')+1:]
    market = csv_path[7:index]
    count = date.count('-')
    if count > 2:
        date = date[:-8]
    df['Date'], df['Day'], df['Market'] = pd.to_datetime(date), day, market
    return df


# read in all files, concat and sort
dfs = [import_csv(csv_path) for csv_path in filenames]
df = pd.concat(dfs, axis=0, ignore_index=True)
df = df.sort_values(['Symbol', 'Date'])

#get rid of  columns with minimal data
# Columns [Div, Yield, P/E,
df = df.drop(['Div', 'Yield', 'P/E'], axis=1)

#fill na's and missing values in entire dataset
df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))

#get list of unique Symbols on the stock market and sort them
symbols = df['Symbol'].unique()
symbols.sort()
#print(df.head())
print("number of different Stock companies:", len(symbols))
#print(symbols)
print("starting Date: ", df['Date'].min())
print("Ending Date: ", df['Date'].max())

#lets try and find the minimun date for each unique symbol
min = df.loc[df['Date'] == df['Date'].min()]
#now maximum date for each unique symbol
max = df.loc[df['Date'] == df['Date'].max()]

diff_max = max[['Symbol', 'Close']].set_index('Symbol')
diff_min = min[['Symbol', 'Open']].set_index('Symbol')
#remove rows not in min and merge
common = diff_max.merge(diff_min, on=['Symbol'])
#print(common.head())
common['ROI'] = common['Close'] - common['Open']
common['Tot % Chg'] = (common['ROI'] / common['Open']) * 100
common = common[['Open', 'Close', 'ROI', 'Tot % Chg']]
common.rename(columns={'Open': 'Open_first_day', 'Close': 'Close_Final_day', 'ROI': 'Total ROI', 'Tot % Chg': 'Tot % Chg'}, inplace=True)
common = common.sort_values('Total ROI', ascending=False)
print()
print("RETURN ON INVESTMENT BY MONEY: ")
print(common.head())

common = common.sort_values("Tot % Chg", ascending=False)
print("RETURN ON INVESTMENT BY % CHANGE:")
print(common.head())
#find values that were deleted in the merge because they don't exist in both DFs
#not_common = diff_max[(~diff_max.index.isin(common.index))]
#print(not_common)

print("I created a dataframe with starting dates and unique Stock Symbols")
print("along with another dataframe with ending dates and unique stock Symbols")
print("Then took the difference of the Open prices on the starting day, and")
print("the Closing prices on the Final day to get a total Return on Investment")
print("This shows: BRKA, AMZN, CMG as the best ROI from start date to end date")
print("This leaves out some Stock companies though, because some companies")
print("do not exist at both the beginning dates and end dates")
print("but this is still useful")
print("Number of stocks left out of this report:", len(symbols) - len(common.index))
#rint(len(symbols) - len(common.index))
