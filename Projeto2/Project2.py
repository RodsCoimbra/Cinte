import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ROI_total(Roi_short_sum, Roi_long_sum):
    return (Roi_short_sum + Roi_long_sum) / 2


def ROI_long(sell, buy):
    return ((sell - buy) / buy) * 100


def ROI_short(sell, buy):
    return ((sell - buy) / sell) * 100


def RSI(df):
    df['RS_7'] = RS(df['Gain'].rolling(7).mean(),
                    df['Loss'].rolling(7).mean())
    df['RS_14'] = RS(df['Gain'].rolling(14).mean(),
                     df['Loss'].rolling(14).mean())
    df['RS_21'] = RS(df['Gain'].rolling(21).mean(),
                     df['Loss'].rolling(21).mean())
    return df


def RS(avg_gain, avg_loss):
    return avg_gain / avg_loss


def ROI_results(df, RSI_short, RSI_long, lll, ull, lls, uls):
    if (RSI_long != 0 and RSI_short != 0 and RSI_short % 7 != 0 and RSI_long % 7 != 0 and RSI_short > 21 and RSI_long > 21
            and lll % 5 != 0 and ull % 5 != 0 and lls % 5 != 0 and uls % 5 != 0
            and lll > 100 and ull > 100 and lls > 100 and uls > 100):
        print("ERRO, parametro mal")
        return
    RSI_period_long = 'RS_' + str(RSI_long)
    RSI_period_short = 'RS_' + str(RSI_short)
    df['RSI_long'] = 100 - (100 / (1 + df[RSI_period_long]))
    df['RSI_short'] = 100 - (100 / (1 + df[RSI_period_short]))
    flag_short = False
    flag_long = False
    Roi_short = []
    Roi_long = []
    sell_short = 0
    buy_long = 0
    for idx, [valueshort, valuelong] in enumerate(zip(df['RSI_short'], df['RSI_long'])):
        if (df['RSI_short'].size == idx + 2):
            if (flag_short):
                buy_short = df['Close'].iloc[idx+1]
                flag_short = False
                Roi_short = np.append(
                    Roi_short, ROI_short(sell_short, buy_short))
            if (flag_long):
                sell_long = df['Close'].iloc[idx+1]
                flag_long = False
                Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
            break

        else:
            if (valueshort > uls and df['RSI_short'].iloc[idx+1] < uls and flag_short == False):
                sell_short = df['Close'].iloc[idx+1]
                flag_short = True
            elif (valueshort > lls and df['RSI_short'].iloc[idx+1] < lls and flag_short == True):
                buy_short = df['Close'].iloc[idx+1]
                flag_short = False
                Roi_short = np.append(
                    Roi_short, ROI_short(sell_short, buy_short))
            if (valuelong < lll and df['RSI_long'].iloc[idx+1] > lll and flag_long == False):
                buy_long = df['Close'].iloc[idx+1]
                flag_long = True
            elif (valuelong < ull and df['RSI_long'].iloc[idx+1] > ull and flag_long == True):
                sell_long = df['Close'].iloc[idx+1]
                flag_long = False
                Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
    return ROI_total(Roi_short.sum(), Roi_long.sum())


if __name__ == '__main__':
    path = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F',
            'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']
    df = {}

    RSI_period_long = 7
    RSI_period_short = 7

    # Read data from csv files
    start = pd.to_datetime('01-01-2020', dayfirst=True)
    end = pd.to_datetime('31-12-2022', dayfirst=True)
    start = pd.to_datetime('01-08-2023', dayfirst=True)
    end = pd.to_datetime('15-09-2023', dayfirst=True)
    for i in path:
        df[i] = pd.read_csv('Data/' + i + '.csv', sep=';', decimal='.',
                            usecols=['Date', 'Close'])
        df[i]['Date'] = pd.to_datetime(df[i]['Date'], dayfirst=True)
        df[i] = df[i][(df[i]['Date'] >= start) & (df[i]['Date'] <= end)]
    # for i in path:
        # a = np.where(df[i]['Close'].shift(1) > df[i]['Close'], 1, 0)
        # df[i]['Gain'] = np.where(
        #     a == 0, df[i]['Close'] - df[i]['Close'].shift(1), 0)
        # df[i]['Loss'] = np.where(
        #     a == 1, df[i]['Close'].shift(1) - df[i]['Close'], 0)
        # avg_gain(df[i], 7)

    for i in path:
        diff = pd.Series(df[i]['Close'].diff())
        df[i]['Gain'] = diff.where(diff > 0, 0)
        df[i]['Loss'] = abs(diff.where(diff < 0, 0))
        df[i] = RSI(df[i])

    print("\n\n", ROI_results(df['AAPL'], RSI_short = 7, RSI_long = 7, lll = 35, ull = 70, lls = 30,  uls =70))
