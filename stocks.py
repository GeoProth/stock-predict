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
    df = pd.read_csv(csv_path)
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


dfs = [import_csv(csv_path) for csv_path in filenames]

df = pd.concat(dfs)



print(df.head())

print(df.tail())
print("shape: ", df.shape)

