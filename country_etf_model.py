import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override() # <== that's all it takes :-)
pd.set_option('precision',4)
pd.options.display.float_format = '{:.3f}'.format


def pull_data(s):
    data_df =  pdr.get_data_yahoo(s, start="2000-11-30", end="2018-04-30")
    return data_df['Open'],data_df['High'], data_df['Low'], data_df['Close'], data_df['Adj Close'], data_df['Volume']

def read_price_file(frq = 'BM'):
    df_price = pd.read_csv("C:/Python27/Git/All_Country_ETF/aclose.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    return df_price

def stock_data_csv(universeList):
    open_data = []
    high_data = []
    low_data = []
    close_data = []
    aclose_data = []
    volume_data = []

    for s in universeList:
        #request OHLC data from yahoo for the universe
        print("****************************** " + s + " ************************")
        dfo, dfh, dfl, dfc, df_ac, dfv = pull_data(s)
        open_data.append(dfo)
        high_data.append(dfh)
        low_data.append(dfl)
        close_data.append(dfc)
        aclose_data.append(df_ac)
        volume_data.append(dfv)

    #concat data for the universe
    open_data = pd.concat(open_data, axis = 1)
    high_data = pd.concat(high_data, axis = 1)
    low_data = pd.concat(low_data, axis = 1)
    close_data = pd.concat(close_data, axis = 1)
    aclose_data = pd.concat(aclose_data, axis = 1)
    volume_data = pd.concat(volume_data, axis = 1)

    # rename columns
    open_data.columns = universe_list
    high_data.columns = universe_list
    low_data.columns = universe_list
    close_data.columns = universe_list
    aclose_data.columns = universe_list
    volume_data.columns = universe_list

    #save the dataframes as csv
    open_data.to_csv("C:/Python27/Git/All_Country_ETF/open.csv")
    high_data.to_csv("C:/Python27/Git/All_Country_ETF/high.csv")
    low_data.to_csv("C:/Python27/Git/All_Country_ETF/low.csv")
    close_data.to_csv("C:/Python27/Git/All_Country_ETF/close.csv")
    aclose_data.to_csv("C:/Python27/Git/All_Country_ETF/aclose.csv")
    volume_data.to_csv("C:/Python27/Git/All_Country_ETF/volume.csv")

def format_oecd_data():
    read_oecd = pd.read_csv("C:/Python27/Git/All_Country_ETF/LEI_Data.csv", index_col= ['TIME'])
    read_oecd = read_oecd[['LOCATION','Value']]
    grouped_oecd = read_oecd.pivot(columns='LOCATION', values='Value')
    read_oecd_info = pd.read_csv("C:/Python27/Git/All_Country_ETF/Universe_Description.csv")
    oecd_info_ls = read_oecd_info.OECD_Codes.tolist()
    oecd_data_ls = grouped_oecd.columns.tolist()
    clean_list = []
    for x in oecd_data_ls:
        if x in oecd_info_ls:

            clean_list.append(x)

    grouped_oecd = grouped_oecd[clean_list]['2000-12':]
    level_change = grouped_oecd.pct_change(periods = 12)
    monthly_change = grouped_oecd.pct_change(periods = 1)
    print(monthly_change)




if __name__ == "__main__":

    universe_list = ['NGE', 'EGPT', 'GXC', 'NORW', 'EPU', 'VNM', 'THD', 'ECH', 'PGAL', 'KSA', 'EWO', 'EWS', 'EWH',
                     'EWJ', 'ENZL', 'EUSA', 'EWI', 'EWQ', 'EWC', 'EWM', 'EIRL', 'EWY', 'EWN', 'EWZ', 'EWA', 'EWT',
                     'EWG', 'EWU', 'EZA', 'EWD', 'EWK', 'PIN', 'GXG', 'PLND', 'EWL', 'EIS', 'EWP', 'GREK', 'UAE', 'QAT',
                     'EPHE', 'EWW', 'IDX', 'TUR']

    # stock_data_csv(universe_list)
    # format_oecd_data()
    px = read_price_file(frq='BM')
    print(px.isnull().sum())