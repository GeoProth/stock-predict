from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import seaborn as sns
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
    df1 = pd.read_csv(csv_path, na_values=missing_values)
    index = csv_path.find('-')
    date = csv_path[index+1:-4]
    day = date[:date.find('-')]
    date = date[date.find('-')+1:]
    market = csv_path[7:index]
    count = date.count('-')
    if count > 2:
        date = date[:-8]
    df1['Date'], df1['Day'], df1['Market'] = pd.to_datetime(date), day, market
    return df1


# read in all files, concat and sort
df1s = [import_csv(csv_path) for csv_path in filenames]
df1 = pd.concat(df1s, axis=0, ignore_index=True)
df1 = df1.sort_values(['Symbol', 'Date'])
df1['Week_Num'] = df1['Date'].dt.week
df1['Day_Num'] = df1['Day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5})
df1['Month'] = df1['Date'].dt.month
#get rid of  columns with minimal data
# Columns [Div, Yield, P/E,
df1 = df1.drop(['Div', 'Yield', 'P/E'], axis=1)

#fill na's and missing values in entire dataset
df1 = df1.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))

#get list of unique Symbols on the stock market and sort them
symbols = df1['Symbol'].unique()
symbols.sort()
stocks = df1.copy()


#------------------------------------------------------------------------------------------
    #this section is for determining the Top 3 Stocks from start date to end date

print("starting Date: ", df1['Date'].min())
print("Ending Date: ", df1['Date'].max())

#print(df1.head())
print("number of different Stock companies:", len(symbols))
#print(symbols)

def stock_eval(df):
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
    common.rename(columns={'Open': 'Open_first_day', 'Close': 'Close_Final_day', 'ROI': '$ Increase', 'Tot % Chg': 'Tot % Chg'}, inplace=True)
    common = common.sort_values('$ Increase', ascending=False)
    
    #amount of buying power after $5 broker fee
    invest = 10000 - 5
    common['$Owned'] = invest
    
    #this column represents how much money is left over after purchasing stocks
    common['$Left'] = common['$Owned'] % common['Open_first_day']
    #this column represents how many shares you can purchase evenly
    common['Shares'] = (common['$Owned'] - common['$Left']) / common['Open_first_day']
    
    #this column represents how much your shares are worth at closing day per share
    #common['$ClosingWorthPerShare'] = (common['Shares'] / (common['Tot % Chg'] / 100))
    #common['$ClosingWorthPerShare'] = round((common['Shares'] + common['$ClosingWorthPerShare']), 2)
    
    common['$TotalProfit'] = (common['Shares'] * common['Close_Final_day']) - 5
    common['$TotalProfit'] = common['$TotalProfit'].map('{:,.2f}'.format)
    return common



print("SORTED BY INCREASE IN STOCK PRICE:")
common = stock_eval(df1)
print(common.head())
print(common.tail())
print()

common = common.sort_values("Tot % Chg", ascending=False)
print("RETURN ON INVESTMENT BY % CHANGE:")
print(common.head())
print(common.tail())
#find values that were deleted in the merge because they don't exist in both df1s
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


#print(stocks.head())
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
#-----------------------------------------------------------------------------------------------

'''

top_3 = df1.loc[(df1['Symbol'] == 'ETSY') | (df1['Symbol'] == 'NFLX') | (df1['Symbol'] == 'AMZN')]

#print(top_3.head())

returns = stock_eval(top_3)
print()
print("RETURN ON INVESTMENT FROM TOP PICKS: ")
#print(returns)

#pick favorite stocks for analysis
etsy = stocks.loc[stocks['Symbol'] == 'ETSY']
ntflx = stocks.loc[stocks['Symbol'] == 'NFLX']
amzn = stocks.loc[stocks['Symbol'] == 'AMZN']
#print(len(etsy.index))


def correct(preds, compare):
    correct = 0
    for x in range(len(compare)):
        value1 = compare[x]
        value2 = preds[x]
        if abs(value1 - value2) <= 1:
            correct += 1
    results = str(correct) + "/" + str(len(compare))
    return results

#split data for training and testing 8/20
def split_data(df1):
    train = df1[:156]
    test = df1[156:]
    response = [train, test]
    return response


train, test = split_data(etsy)

#print(train.head())
pd.set_option('display.precision', 4)
correlation = train[['Close', 'Month', 'Day_Num', 'Week_Num', 'Open', 'High', 'Low', 'Volume', '52 Wk Low', '52 Wk High']]
corr = correlation.corr()
'''
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.figure(figsize=(12, 6))
plt.title("Correlation Matrix to Closing Price")
plt.show()
'''
X_train = train.drop(['Symbol', 'Close'], axis=1)
Y_train = train['Close']
X_test = test.drop(['Symbol', 'Close'], axis=1)
Y_test = test['Close']


#----------------------------------------------------------
    #implement Linear Regression
lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)

test['Predictions'] = lr.predict(X_test)

#print(preds)
compare = list(Y_test)
#print(compare)

# ADD HOLT WINTERS SMOOTHING
#print(test.head())
model = ExponentialSmoothing(train['Close'], seasonal_periods=114, seasonal='add').fit()
pred = list(model.predict(start=len(train), end=195))

print(pred)
print(len(pred))
test['Holt_Winter'] = pred

train['Close'].plot(label='train data', figsize=(14, 6), title='ETSY')
test['Close'].plot(label='Actual Closing price')
test['Predictions'].plot(label='Linear Regression Preds')
#test['Holt_Winter'].plot(label='Holt-Winter Pred')
test['Holt_Winter'].plot(label='Holt_Winter Preds')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend(loc='best')
plt.show()


#---------------------------------------------------------------------------------------
    #AMAZON

train, test = split_data(amzn)

X_train = train.drop(['Symbol', 'Close'], axis=1)
Y_train = train['Close']
X_test = test.drop(['Symbol', 'Close'], axis=1)
Y_test = test['Close']

#-----------------------------------------------------------
    #implement Linear Regression
lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)

preds = list(lr.predict(X_test))
#print(preds)

model = ExponentialSmoothing(train['Close'], seasonal_periods=36, seasonal='add').fit()
pred = list(model.predict(start=len(train), end=195))

#Plot
#print(preds)
test['Predictions'] = preds
test['Holt_Winter'] = pred
#print(test.head())

train['Close'].plot(label='train data', figsize=(14, 6), title='AMAZON')
test['Close'].plot(label='Actual Closing price')
test['Predictions'].plot(label='Linear Regression Preds')
test['Holt_Winter'].plot(label='Holt Winters Preds')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------
    #NETFLIX

train, test = split_data(ntflx)

X_train = train.drop(['Symbol', 'Close'], axis=1)
Y_train = train['Close']
X_test = test.drop(['Symbol', 'Close'], axis=1)
Y_test = test['Close']

#--------------------------------------------------------------------------------------------
    #implement Linear Regression
lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)

preds = list(lr.predict(X_test))
#print(preds)

#Holt Winters
model = ExponentialSmoothing(train['Close'], seasonal_periods=36, seasonal='add').fit()
pred = list(model.predict(start=len(train), end=195))

#Plot
#print(preds)
test['Predictions'] = preds
test['Holt_Winter'] = pred
#print(test.head())

train['Close'].plot(label='train data', figsize=(14, 6), title='NETFLIX')
test['Close'].plot(label='Actual Closing price')
test['Predictions'].plot(label='Linear Regression Preds')
test['Holt_Winter'].plot(label='Holt_Winter Preds')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()

#============================================================================================
#X_train = X_train.astype(int)
#Y_train = Y_train.astype(int)
'''

dt = DecisionTreeClassifier(min_samples_split=10, random_state=1)
dt.fit(X_train, Y_train)

X_test = X_test.astype(int)
predict = list(dt.predict(X_test))

results = correct(predict, compare)
print("Decision Tree Preds: ", results)

test['Predictions'] = predict
train['Close'].plot(label='train(Etsy)', figsize=(14, 6), title='Decision Tree (ETSY)')
test['Close'].plot(label='Actual Closing price')
test['Predictions'].plot(label='Predicted Closing price')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()

#---------------------------------------------------------------------------------------

 # Neural Networks LSTM

scaler = StandardScaler()
scaler.fit(X_train)

#Apply Transformations to data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20))

mlp.fit(X_train, Y_train)

predict = list(mlp.predict(X_test))
print(type(predict))
print(predict)

results = correct(predict, compare)
print("Neural Nets Preds: ", results)

test['Predictions'] = predict
train['Close'].plot(label='train(Etsy)', figsize=(14, 6), title='Neural Net (ETSY)')
test['Close'].plot(label='Actual Closing price')
test['Predictions'].plot(label='Predicted Closing price')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()
'''
