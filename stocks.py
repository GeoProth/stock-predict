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
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
df['Week_Num'] = df['Date'].dt.week
df['Day_Num'] = df['Day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5})
df['Month'] = df['Date'].dt.month
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

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

#stocks is our new data set
#symbols is a list of unique symbols for stocks
stocks = stocks.sort_values('Date')
#reorganize this for better visualization
stocks = stocks[['Date', 'Symbol', 'Month', 'Day_Num', 'Week_Num', 'Open', 'High', 'Low', 'Volume',
               '52 Wk Low', '52 Wk High', 'Close']]

#change ints to floats for analysis
stocks['Month'] = stocks['Month'].astype(float)
stocks['Day_Num'] = stocks['Day_Num'].astype(float)
stocks['Week_Num'] = stocks['Week_Num'].astype(float)
#set Date as index
stocks.index = stocks['Date']
stocks = stocks.drop(['Date'], axis=1)
#convert Volume to int
stocks['Volume'] = stocks['Volume'].str.replace(',', '')
stocks['Volume'] =stocks['Volume'].astype(float)

print(stocks.head())
#Separate Data Frames that represent top 3 stocks
sdlr = stocks.loc[stocks['Symbol'] == 'SDRL']
epix = stocks.loc[stocks['Symbol'] == 'EPIX']
hear = stocks.loc[stocks['Symbol'] == 'HEAR']

'''
#plot these 3 stock closing prices from start to finish
sdlr['Close'].plot(label='SDLR', figsize=(14, 6), title='Closing Prices')
epix['Close'].plot(label='EPIX')
hear['Close'].plot(label='HEAR')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


#plot these 3 stock volumes from start to finish

sdlr['Volume'].plot(label='SDLR', figsize=(14, 6), title='Volume Traded')
epix['Volume'].plot(label='EPIX')
hear['Volume'].plot(label='HEAR')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()
'''

#pick favorite stocks for analysis
etsy = stocks.loc[stocks['Symbol'] == 'ETSY']
ntflx = stocks.loc[stocks['Symbol'] == 'NFLX']
amzn = stocks.loc[stocks['Symbol'] == 'AMZN']
print(len(etsy.index))

features = ['Month', 'Day_Num', 'Week_Num', 'Open', 'High', 'Low', 'Volume', '52 Wk Low', '52 Wk High']

#split data for training and testing 8/20
def split_data(df):
    train = df[:156]
    test = df[156:]
    response = [train, test]
    return response

train, test = split_data(etsy)

X_train = train.drop(['Symbol', 'Close'], axis=1)
Y_train = train['Close']
X_test = test.drop(['Symbol', 'Close'], axis=1)
Y_test = test['Close']

#implement Linear Regression
lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)

preds = list(lr.predict(X_test))
#print(preds)
compare = list(Y_test)
#print(compare)

correct = 0
for x in range(len(compare)):
    value1 = compare[x]
    value2 = preds[x]
    if(abs(value1 - value2) <= 1):
        correct += 1

results = str(correct) + "/" + str(len(compare))
print("Linear regression Preds: ", results)

#Plot
print(preds)
test['Predictions'] = preds.copy()
print(test.head())

train['Close'].plot(label='train(Etsy)', figsize=(14, 6), title='Linear Regression (ETSY)')
test['Close'].plot(label='test(Etsy)')
test['Predictions'].plot(label='Predictions')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()
