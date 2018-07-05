import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override() # <== that's all it takes :-)
from scipy import stats
import matplotlib.pyplot as plt
pd.set_option('precision',4)
pd.options.display.float_format = '{:.3f}'.format


def pull_data(s):
    data_df =  pdr.get_data_yahoo(s, start="2000-11-30", end="2018-05-31")
    return data_df['Open'],data_df['High'], data_df['Low'], data_df['Close'], data_df['Adj Close'], data_df['Volume']

def read_price_file(frq = 'BM', cols=[]):
    df_price = pd.read_csv("C:/Python27/Git/All_Country_ETF/aclose.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    df_price = df_price[cols]
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
    #Reading the LEI csv file
    read_oecd = pd.read_csv("C:/Python27/Git/All_Country_ETF/LEI_Data.csv", index_col= ['TIME'])
    read_oecd = read_oecd[['LOCATION','Value']]
    grouped_oecd = read_oecd.pivot(columns='LOCATION', values='Value')

    #The LEI indicator is lagged by 2 months. Adding 2 months on the index and shiifting the data forward
    grouped_oecd.loc['2018-04'] = 0.0
    grouped_oecd.loc['2018-05'] = 0.0
    grouped_oecd = grouped_oecd.shift(2)

    #reading the OECD Description file for mapping symbols and OECD codes
    read_oecd_info = pd.read_csv("C:/Python27/Git/All_Country_ETF/Universe_Description.csv")
    #list of OECD_Codes
    oecd_info_ls = read_oecd_info.OECD_Codes.tolist()
    #List of pivoted OCED_Columns (tickers)
    oecd_data_ls = grouped_oecd.columns.tolist()
    clean_list = []
    #Matching the OECD Codes with the Symbol
    for x in oecd_data_ls:
        if x in oecd_info_ls:
            clean_list.append(x)
    oecd_info_sym = read_oecd_info.Symbol.tolist()
    #truncating the data from 2000
    grouped_oecd = grouped_oecd[clean_list]['2000-12':]
    #mapping the OECD codes to Symbol for the final dataframe
    list_to_dict = dict(zip(oecd_info_sym, oecd_info_ls))
    #Creating the dataframe with the mapped OECD_Codes
    for k, v in list_to_dict.items():
        if v in grouped_oecd.columns:
            grouped_oecd.rename(columns={v: k}, inplace=True)

    #calculating 12 month OECD LEI level change
    level_change = grouped_oecd.pct_change(periods = 12)
    # calculating monthly change of levels
    monthly_change = level_change.pct_change(periods = 3)
    return level_change, monthly_change


def zscore_data(data, window = 12, axis = 0):

    # time series scores for the rolling period
    ts_score = (data - data.rolling(window).mean())/data.rolling(window).std()
    #cliping scores for lower and upper boundary of -3.5 to 3.5
    tsScore = ts_score.clip(-3.5, 3.5)
    # crossectional zscore
    csScore = pd.DataFrame(stats.zscore(tsScore, axis = 1), index = data.index, columns=data.columns)
    # cliping scores for lower and upper boundary of -3.5 to 3.5
    csScore = csScore.clip(-3.5, 3.5)
    return csScore

def fractile_filter(data):
    q_list = []
    for row in data.iterrows():
        low_cond = row[1]>=row[1]['low_filter']
        up_cond = row[1]< row[1]['up_filter']
        cond = (low_cond & up_cond)
        q_list.append(cond)
        # q_list.append(row[1]>row[1]['filter'])
    return q_list

def fractile_analysis(data, px_data, q1, q2):
    #filter data based on the lower and upper bound
    data['low_filter'] = data.quantile(q=q1, numeric_only=True, axis=1)
    data['up_filter'] = data.quantile(q=q2, numeric_only=True, axis=1)
    #dataframe with the filtered data
    data = data[pd.DataFrame(fractile_filter(data))]
    #remove the filter column
    data = data.drop('low_filter', axis = 1)
    data = data.drop('up_filter', axis=1)
    #Calcaulte the monthly price returns and lag it by month
    returns_df  = px_data.pct_change().shift(-1)
    #calculate the returns of the holding for the fractiles
    fractile_return = returns_df[data.notnull()]
    ic_sig = data.corrwith(fractile_return, axis = 1, drop=True)
    fractile_return= fractile_return.shift(1)
    #prints the trade recommendations
    print("Trades for %s decile is" %(str(q1)))
    print(fractile_return[-1:].dropna(axis= 1))
    return fractile_return, ic_sig

def signal_test(scores, prices, q):
    sc_filter= scores>=scores.quantile(q, axis=1)
    filtered_scores = scores[sc_filter]
    returns_df = prices.pct_change().shift(-1)
    returns_df = returns_df[sc_filter]
    ts_corr = filtered_scores.corrwith(returns_df,axis = 1,drop=True).dropna()
    return ts_corr

if __name__ == "__main__":

    universe_list = ['NGE', 'EGPT', 'GXC', 'NORW', 'EPU', 'VNM', 'THD', 'ECH', 'PGAL', 'KSA', 'EWO', 'EWS', 'EWH',
                     'EWJ', 'ENZL', 'EUSA', 'EWI', 'EWQ', 'EWC', 'EWM', 'EIRL', 'EWY', 'EWN', 'EWZ', 'EWA', 'EWT',
                     'EWG', 'EWU', 'EZA', 'EWD', 'EWK', 'PIN', 'GXG', 'PLND', 'EWL', 'EIS', 'EWP', 'GREK', 'UAE', 'QAT',
                     'EPHE', 'EWW', 'IDX', 'TUR']

    #pull historical data
    # stock_data_csv(universe_list)
    levelChange, monthlyChange = format_oecd_data()
    price_df = read_price_file(frq='BM', cols = levelChange.columns)
    #crossectional scores for 12 months of level change
    cs_Score_level = zscore_data(levelChange, window = 3, axis = 1)
    #cross-sectional scores for i month of level change
    cs_Score_monthly = pd.DataFrame(stats.zscore(monthlyChange, axis = 1), index = monthlyChange.index, columns=monthlyChange.columns)
    # cliping scores for lower and upper boundary of -3.5 to 3.5
    cs_Score_monthly = cs_Score_monthly.clip(-3.5, 3.5)
    #composite the level change factor scores
    composite_oecd_score = (1.0*cs_Score_level) + (0.0*cs_Score_monthly)
    #adjusting the date index to year and month format to match the price data
    price_df.index = price_df.index.strftime("%Y-%m")
    #list for fractile analysis
    fractile_list = np.arange(0.0, 1.0, 0.1)
    #looping ove to get the fractile return dataframe
    returns_list = []
    ic_list = []
    for i in fractile_list:

        frac_portfolio,ic_portfolio = fractile_analysis(composite_oecd_score, price_df, q1 =i, q2 = i+0.1)
        temp = frac_portfolio.mean(axis=1).values.tolist()
        returns_list.append(temp)
        ic_list.append(ic_portfolio)

    fractile_df = pd.DataFrame({i :returns_list[i] for i in range(len(returns_list))}, index=price_df.index)
    fractile_df = fractile_df.rename(columns=({0:'Q1', 1:'Q2',2:'Q3', 3:'Q4', 4:'Q5',5:'Q6', 6:'Q7',7:'Q8', 8:'Q9',9:'Q10'}))
    fractile_df.dropna(inplace=True)
    fractile_spread =  fractile_df['Q10'].mean() - fractile_df.mean()
    fractile_df.loc['2005-07'] = 0.0
    #calculate the decile
    ic_df = pd.DataFrame(ic_list)
    ic_df = ic_df.T
    ic_df = ic_df.rename(columns=({0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4', 4: 'Q5', 5: 'Q6', 6: 'Q7', 7: 'Q8', 8: 'Q9', 9: 'Q10'}))
    ic_df['Q10'].rolling(6).mean().plot(kind = 'bar')
    plt.grid()
    plt.show()











