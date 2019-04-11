from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
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
df['Week_Number'] = df['Date'].dt.week
df['Day'] = df['Day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5})
#get rid of  columns with minimal data
# Columns [Div, Yield, P/E,
df = df.drop(['Div', 'Yield', 'P/E'], axis=1)

#fill na's and missing values in entire dataset
df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))

#get list of unique Symbols on the stock market and sort them
symbols = df['Symbol'].unique()
symbols.sort()
stocks = df.copy()

#------------------------------------------------------------------------------------------
    #this section is for determining the Top 3 Stocks from start date to end date

print("starting Date: ", df['Date'].min())
print("Ending Date: ", df['Date'].max())

#print(df.head())
print("number of different Stock companies:", len(symbols))
#print(symbols)


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
#total amount available to invest = 10000 - 5 for broker fee
common = common[['Open', 'Close', 'ROI', 'Tot % Chg']]
common.rename(columns={'Open': 'Open_first_day', 'Close': 'Close_Final_day', 'ROI': 'Total ROI', 'Tot % Chg': 'Tot % Chg'}, inplace=True)
common = common.sort_values('Total ROI', ascending=False)

#amount of buying power after $5 broker fee
invest = 10000 - 5
common['$Owned'] = invest

#this column represents how much money is left over after purchasing stocks
common['$Left'] = common['$Owned'] % common['Open_first_day']
#this column represents how many shares you can purchase evenly
common['Shares'] = (common['$Owned'] - common['$Left']) / common['Open_first_day']

#this column represents how much your shares are worth at closing day per share
common['$ClosingWorthPerShare'] = (common['Shares'] / (common['Tot % Chg'] / 100))
common['$ClosingWorthPerShare'] = round((common['Shares'] + common['$ClosingWorthPerShare']), 2)

common['$TotalProfit'] = (common['Shares'] * common['$ClosingWorthPerShare']) - 5
common['$TotalProfit'] = common['$TotalProfit'].map('{:,.2f}'.format)

print()

common = common.sort_values("Tot % Chg", ascending=False)
print("RETURN ON INVESTMENT BY % CHANGE:")
print(common.head())
print(common.tail())
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


#------------------------------------------------------------------------------------------------------------------

    #This section is for prediction algorithms
'''
#stocks is our new data set
#symbols is a list of unique symbols for stocks
stocks = stocks.sort_values('Date')
stocks = stocks.drop(['YTD % Chg', '% Chg', 'Name'], axis=1)
stocks['Month'] = stocks['Date'].dt.month
stocks = stocks[['Market', 'Symbol', 'Day', 'Month', 'Date', 'Week_Number', 'Open', 'High', 'Low', 'Volume',
                 '52 Wk High', '52 Wk Low', 'Close']]

print(stocks.head(10))

df = pd.pivot_table(stocks, index=['Symbol', 'Date'])

print(df.head())

print(df.tail())

df = df.sort_values('Date')

print(df.head(50))

y = stocks[['Close']]
stocks = stocks.drop(['Date', 'Market', 'Close', 'Volume'], axis=1)

#stocks = pd.get_dummies(data=stocks, columns=['Symbol'])



print(stocks.head())
print(stocks.shape)
print(y.head())

X = stocks.copy().astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=0)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = list(lm.predict(X_test))

y_test_list = list(y_test)

correct = 0
for x in range(len(predictions)):
    value1 = predictions[x]
    value2 = y_test_list[x]
    if(abs(value1 - value2) <= 3):
        correct += 1

results = str(correct) + "/" + str(len(predictions))
print("Linear regression: ", results)
#y_train = stocks[[]]
'''